package org.broadinstitute.hellbender.tools.sv;

import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.broadinstitute.hellbender.utils.QualityUtils;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.*;

public class EvidenceStatUtils {

    private static final Median MEDIAN = new Median();

    /**
     * Performs refinement on one side of a breakpoint
     *
     * @param evidence split read evidence to test
     * @param carrierSamples carrier sample ids
     * @param backgroundSamples background sample ids
     * @param defaultPosition position to use if test cannot be performed (no evidence or carriers)
     * @return pair containing site with refined breakpoints and probability (null if no evidence or carriers)
     */
    public static SplitReadSite refineSplitReadSite(final List<SplitReadEvidence> evidence,
                                                    final Set<String> carrierSamples,
                                                    final Set<String> backgroundSamples,
                                                    final Map<String, Double> sampleCoverageMap,
                                                    final int representativeDepth,
                                                    final int defaultPosition) {
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(carrierSamples),
                "One or more carrier samples not found in sample coverage map");
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(backgroundSamples),
                "One or more non-carrier samples not found in sample coverage map");

        // Default case
        if (evidence.isEmpty() || carrierSamples.isEmpty()) {
            return new SplitReadSite(defaultPosition, Collections.emptyMap(), null);
        }

        Double minP = null;
        Integer minDistance = null;
        Integer minPosition = null;
        Map<String,Integer> minSampleCounts = null;
        int position = 0;
        Map<String,Integer> sampleCounts = new HashMap<>();
        for (final SplitReadEvidence e : evidence) {
            if (e.getStart() != position) {
                final double p = EvidenceStatUtils.calculateOneSamplePoissonTest(
                        sampleCounts, carrierSamples, backgroundSamples, sampleCoverageMap, representativeDepth);
                if (minP == null || p <= minP) {
                    final int dist = Math.abs(position - defaultPosition);
                    if (minPosition == null || dist < minDistance) {
                        minP = p;
                        minPosition = position;
                        minDistance = dist;
                        minSampleCounts = sampleCounts;
                    }
                }
                sampleCounts = new HashMap<>();
                position = e.getStart();
            }
            sampleCounts.put(e.getSample(), e.getCount());
        }
        return new SplitReadSite(minPosition, minSampleCounts, minP);
    }

    public static Integer probToQual(final Double pError, final byte maxQual) {
        return pError == null ? null : (int) QualityUtils.errorProbToQual(pError, maxQual);
    }

    public static int computeRepresentativeDepth(final Collection<Double> coverageValues) {
        double meanCoverage = 0;
        for (final Double c : coverageValues) {
            meanCoverage += c;
        }
        return (int) Math.round(meanCoverage / coverageValues.size());
    }

    /**
     * Find the median of site counts in a subset of samples when normalized by sample coverage
     *
     * @param samples sample ids to restrict to
     * @param sampleCounts
     * @return median
     */
    @VisibleForTesting
    protected static double getMedianNormalizedCount(final Set<String> samples,
                                                  final Map<String, Integer> sampleCounts,
                                                  final Map<String, Double> sampleCoverageMap) {
        final double[] normalizedCounts = new double[samples.size()];
        int i = 0;
        for (final String sample : samples) {
            normalizedCounts[i] = sampleCounts.getOrDefault(sample, 0) / sampleCoverageMap.get(sample);
            i++;
        }
        return MEDIAN.evaluate(normalizedCounts);
    }

    public static double cumulativePoissonProbability(final double mean, final int x) {
        if (x < 0) {
            return 0;
        }
        if (x == Integer.MAX_VALUE) {
            return 1;
        }
        return Gamma.regularizedGammaQ((double) x + 1, mean);
    }

    /**
     * Performs poisson test on a single site by computing the probability of observing the background counts
     * under a carrier count distribution
     *
     * @param sampleCounts
     * @param carrierSamples
     * @param backgroundSamples
     * @param meanCoverage mean coverage of all samples
     * @return probability on [0,1]
     */
    public static double calculateOneSamplePoissonTest(final Map<String, Integer> sampleCounts,
                                                       final Set<String> carrierSamples,
                                                       final Set<String> backgroundSamples,
                                                       final Map<String, Double> sampleCoverageMap,
                                                       final double meanCoverage) {
        final double medianNormalizedCarrierCount = EvidenceStatUtils.getMedianNormalizedCount(carrierSamples, sampleCounts, sampleCoverageMap);
        if (medianNormalizedCarrierCount == 0) {
            return 1.;  // Degenerate case in which the Poisson distribution is undefined
        }
        final double medianBackgroundRate = EvidenceStatUtils.getMedianNormalizedCount(backgroundSamples, sampleCounts, sampleCoverageMap);
        final int backgroundCount = (int) Math.round(medianBackgroundRate * meanCoverage);
        return EvidenceStatUtils.cumulativePoissonProbability(meanCoverage * medianNormalizedCarrierCount, backgroundCount);
    }
}
