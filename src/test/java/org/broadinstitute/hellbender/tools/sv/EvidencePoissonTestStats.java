package org.broadinstitute.hellbender.tools.sv;

import org.apache.commons.math3.distribution.PoissonDistribution;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class EvidencePoissonTestStats {

    private static final double ERROR_TOL = 1e-6;

    @DataProvider(name = "testCumulativePoissonProbabilityData")
    public Object[][] testCumulativePoissonProbabilityData() {
        return new Object[][]{
                {1.0, 0},
                {1.0, 1},
                {1.0, 2},
                {1.0, 3},
                {1.0, 10},
                {1.0, 100},
                {2.0, 0},
                {2.0, 1},
                {2.0, 2},
                {2.0, 3},
                {2.0, 10},
                {2.0, 100}
        };
    }

    @Test(dataProvider= "testCumulativePoissonProbabilityData")
    public void testCumulativePoissonProbability(final double lambda, final int val) {
        final double expected = new PoissonDistribution(lambda).cumulativeProbability(val);
        final double test = EvidenceStatUtils.cumulativePoissonProbability(lambda, val);
        Assert.assertTrue(Math.abs(test - expected) <= ERROR_TOL);
    }

    @DataProvider(name = "testGetMedianNormalizedCountData")
    public Object[][] testGetMedianNormalizedCountData() {
        return new Object[][]{
                {new double[]{30.}, new int[]{0}, 0},
                {new double[]{30.}, new int[]{1}, 1/30.},
                {new double[]{30., 30.}, new int[]{0, 0}, 0},
                {new double[]{30., 30.}, new int[]{1, 1}, 1/30.},
                {new double[]{30., 45.}, new int[]{1, 1}, 0.5*((1/30.)+(1/45.))},
                {new double[]{30., 30., 30.}, new int[]{1, 0, 2}, 1/30.},
                {new double[]{30., 30., 30.}, new int[]{0, 1, 2}, 1/30.},
                {new double[]{15., 30., 45.}, new int[]{1, 1, 1}, 1/30.},
                {new double[]{16., 30., 45.}, new int[]{1, 2, 2}, 1/16.},
        };
    }

    @Test(dataProvider= "testGetMedianNormalizedCountData")
    public void testGetMedianNormalizedCount(final double[] sampleCoverage, final int[] sampleCounts, final double expected) {
        final Map<String, Double> sampleCoverageMap = new HashMap<>();
        final Map<String, Integer> sampleCountsMap = new HashMap<>();
        final Set<String> samples = new HashSet<>();
        for (int i = 0; i < sampleCoverage.length; i++) {
            final String sample = "sample" + i;
            sampleCoverageMap.put(sample, sampleCoverage[i]);
            if (sampleCounts[i] > 0) {
                sampleCountsMap.put(sample, sampleCounts[i]);
            }
            samples.add(sample);
        }

        final double test = EvidenceStatUtils.getMedianNormalizedCount(samples, sampleCountsMap, sampleCoverageMap);
        Assert.assertTrue(Math.abs(test - expected) <= ERROR_TOL);
    }
}