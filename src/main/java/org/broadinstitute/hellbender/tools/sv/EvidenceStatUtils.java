package org.broadinstitute.hellbender.tools.sv;

import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.stat.descriptive.rank.Median;
import org.broadinstitute.hellbender.utils.QualityUtils;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.Collection;
import java.util.Map;

public class EvidenceStatUtils {

    private static final Median MEDIAN = new Median();

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
    protected static double getMedianNormalizedCount(final Collection<String> samples,
                                                     final Map<String, Integer> sampleCounts,
                                                     final Map<String, Double> sampleCoverageMap) {
        if (samples.isEmpty()) {
            return 0;
        }
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
                                                       final Collection<String> carrierSamples,
                                                       final Collection<String> backgroundSamples,
                                                       final Map<String, Double> sampleCoverageMap,
                                                       final double meanCoverage,
                                                       final double expectedBackground) {
        Utils.validateArg(!sampleCounts.isEmpty(), "Sample counts required");
        final double medianNormalizedCarrierCount = expectedBackground + EvidenceStatUtils.getMedianNormalizedCount(carrierSamples, sampleCounts, sampleCoverageMap);
        final double medianBackgroundRate = EvidenceStatUtils.getMedianNormalizedCount(backgroundSamples, sampleCounts, sampleCoverageMap);
        final int backgroundCount = (int) Math.round(medianBackgroundRate * meanCoverage);
        return EvidenceStatUtils.cumulativePoissonProbability(meanCoverage * medianNormalizedCarrierCount, backgroundCount);
    }
}
