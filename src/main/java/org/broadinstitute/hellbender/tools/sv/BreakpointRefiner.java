package org.broadinstitute.hellbender.tools.sv;

import com.google.common.collect.Sets;
import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.variant.variantcontext.StructuralVariantType;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Refines variant breakpoints using split read evidence.
 *
 * The start and end of the breakpoint are tested independently. At each end we perform a series of Poisson
 * tests using the following model:
 *
 *  Given:
 *      d_i : depth of sample i
 *      c_i : split read count of sample i
 *
 *   Define a Poisson model of split read counts:
 *      r_i = c_i / d_i : normalized split read count of sample i
 *      m_c : median carrier r_i
 *      m_b : median background (non-carrier) r_i
 *      mu : mean depth of all samples
 *
 *      lambda = mu * m_c : expected carrier count
 *      X ~ Poisson(lambda) : carrier count model
 *      x_b = round(mu * m_b) : adjusted median background count
 *
 *   Calculate probability of observing the background count:
 *      p = P(X < x_b)
 *
 *   We then select the site with the lowest score. Breakpoint end positions are restricted by a lowerbound that depends
 *   on the refined start position (see {@link BreakpointRefiner#getEndLowerBound(SVCallRecord, int)}.
 */
public class BreakpointRefiner {

    private final Map<String,Double> sampleCoverageMap;
    private final SAMSequenceDictionary dictionary;
    /**
     * Number bases that insertion split read positions can pass by the original breakpoint, when left-clipped
     * reads are to the left of the breakpoint and right-clipped reads to the right.
     */
    private int maxInsertionSplitReadCrossDistance;

    private static final Median MEDIAN = new Median();

    public static final int DEFAULT_MAX_INSERTION_CROSS_DISTANCE = 20;

    /**
     * @param sampleCoverageMap map with (sample id, per-base sample coverage) entries
     * @param dictionary reference dictionary
     */
    public BreakpointRefiner(final Map<String,Double> sampleCoverageMap, final SAMSequenceDictionary dictionary) {
        this.sampleCoverageMap = Utils.nonNull(sampleCoverageMap);
        this.dictionary = Utils.nonNull(dictionary);
        setMaxInsertionSplitReadCrossDistance(DEFAULT_MAX_INSERTION_CROSS_DISTANCE);
    }

    public void setMaxInsertionSplitReadCrossDistance(final int distance) {
        maxInsertionSplitReadCrossDistance = distance;
    }

    /**
     * Performs breakend refinement for a call
     *
     * @param record with split read evidence
     * @return record with new breakpoints
     */
    public SVCallRecord refineCall(final SVCallRecord record, final List<SplitReadSite> startSplitReadSites,
                                   final List<SplitReadSite> endSplitReadSites) {
        Utils.nonNull(record);
        SVCallRecordUtils.validateCoordinatesWithDictionary(record, dictionary);

        // Depth-only calls cannot be refined
        if (record.isDepthOnly()) {
            return record;
        }

        // Sample sets
        final Set<String> callSamples = record.getAllSamples();
        final Set<String> calledSamples = record.getCarrierSampleSet();
        final Set<String> backgroundSamples = Sets.difference(callSamples, calledSamples);
        final int representativeDepth = computeRepresentativeDepth(callSamples);

        // Refine start
        final SplitReadSite refinedStartSite = getRefinedSite(startSplitReadSites, calledSamples,
                backgroundSamples, representativeDepth, record.getPositionA());

        // Refine end
        final int endLowerBound = getEndLowerBound(record, refinedStartSite.getPosition());
        final int defaultEndPosition = Math.max(endLowerBound, record.getPositionB());
        final List<SplitReadSite> validEndSites = getValidEndSplitReadSites(endSplitReadSites, endLowerBound);
        final SplitReadSite refinedEndSite = getRefinedSite(validEndSites, calledSamples, backgroundSamples,
                representativeDepth, defaultEndPosition);

        final int refinedStartPosition = refinedStartSite.getPosition();
        final int refinedEndPosition = refinedEndSite.getPosition();
        final Integer length = record.getType().equals(StructuralVariantType.INS) ? record.getLength() : null;

        // Create new record
        return new SVCallRecord(record.getId(), record.getContigA(), refinedStartPosition,
                record.getStrandA(), record.getContigB(), refinedEndPosition, record.getStrandB(), record.getType(),
                length, record.getAlgorithms(), record.getAlleles(), record.getGenotypes(), record.getAttributes());
    }

    /**
     * Filters end sites with position less than lower-bound
     *
     * @param endSplitReadSites
     * @param lowerBound min position
     * @return filtered set of end sites
     */
    private List<SplitReadSite> getValidEndSplitReadSites(final List<SplitReadSite> endSplitReadSites, final int lowerBound) {
        return endSplitReadSites.stream().filter(s -> s.getPosition() >= lowerBound).collect(Collectors.toList());
    }

    /**
     * Determines lower-bound on end site position (inclusive). For inter-chromosomal variants, boundaries are at the
     * start of the chromsome (any position is valid). For insertions, {@link BreakpointRefiner#maxInsertionSplitReadCrossDistance}
     * is used to determine how far past the original breakpoint it can be. Otherwise, we just use the new start position.
     *
     * @param call
     * @param refinedStartPosition new start position of call
     * @return position
     */
    private int getEndLowerBound(final SVCallRecord call, final int refinedStartPosition) {
        if (!call.isIntrachromosomal()) {
            return 1;
        }
        if (call.getType().equals(StructuralVariantType.INS)) {
            return refinedStartPosition - maxInsertionSplitReadCrossDistance;
        }
        return refinedStartPosition + 1;
    }

    private int computeRepresentativeDepth(final Collection<String> samples) {
        double meanCoverage = 0;
        for (final String s : samples) {
            meanCoverage += sampleCoverageMap.get(s);
        }
        return (int) Math.round(meanCoverage / samples.size());
    }

    /**
     * Performs refinement on one side of a breakpoint
     *
     * @param sites sites to test
     * @param carrierSamples carrier sample ids
     * @param backgroundSamples background sample ids
     * @param defaultPosition position to use if test cannot be performed (no sites or no carriers)
     * @return site with refined breakpoints
     */
    private SplitReadSite getRefinedSite(final List<SplitReadSite> sites,
                                         final Set<String> carrierSamples,
                                         final Set<String> backgroundSamples,
                                         final int representativeDepth,
                                         final int defaultPosition) {
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(carrierSamples),
                "One or more carrier samples not found in sample coverage map");
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(backgroundSamples),
                "One or more non-carrier samples not found in sample coverage map");

        // Default case
        if (sites.isEmpty() || carrierSamples.isEmpty()) return new SplitReadSite(defaultPosition, Collections.emptyMap());

        Double minLogP = null;
        Integer minDistance = null;
        SplitReadSite minSite = null;
        for (final SplitReadSite s : sites) {
            final double logP = Double.valueOf(calculateOneSamplePoissonTest(s, carrierSamples, backgroundSamples, representativeDepth));
            if (minLogP == null || logP <= minLogP) {
                final int dist = Math.abs(s.getPosition() - defaultPosition);
                if (minSite == null || dist < minDistance) {
                    minLogP = logP;
                    minSite = s;
                    minDistance = dist;
                }
            }
        }
        return minSite;
    }

    /**
     * Performs poisson test on a single site by computing the probability of observing the background counts
     * under a carrier count distribution
     *
     * @param site
     * @param carrierSamples
     * @param backgroundSamples
     * @param meanCoverage mean coverage of all samples
     * @return probability
     */
    private double calculateOneSamplePoissonTest(final SplitReadSite site,
                                                 final Set<String> carrierSamples,
                                                 final Set<String> backgroundSamples,
                                                 final double meanCoverage) {
        final double medianNormalizedCarrierCount = getMedianNormalizedCount(carrierSamples, site);
        if (medianNormalizedCarrierCount == 0) {
            return 1;  // Degenerate case in which the Poisson distribution is undefined
        }
        final double medianBackgroundRate = getMedianNormalizedCount(backgroundSamples, site);
        final int backgroundCount = (int) Math.round(medianBackgroundRate * meanCoverage);
        return cumulativePoissonProbability(meanCoverage * medianNormalizedCarrierCount, backgroundCount);
    }

    /**
     * Find the median of site counts in a subset of samples when normalized by sample coverage
     *
     * @param samples sample ids to restrict to
     * @param site
     * @return median
     */
    private double getMedianNormalizedCount(final Set<String> samples, final SplitReadSite site) {
        final double[] normalizedCounts = new double[samples.size()];
        int i = 0;
        for (final String sample : samples) {
            normalizedCounts[i] = site.getCount(sample) / sampleCoverageMap.get(sample);
        }
        return MEDIAN.evaluate(normalizedCounts);
    }

    private static double cumulativePoissonProbability(final double mean, final int x) {
        if (x < 0) {
            return 0;
        }
        if (x == Integer.MAX_VALUE) {
            return 1;
        }
        return Gamma.regularizedGammaQ((double) x + 1, mean);
    }
}
