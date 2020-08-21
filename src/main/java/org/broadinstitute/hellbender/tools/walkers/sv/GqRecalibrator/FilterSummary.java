package org.broadinstitute.hellbender.tools.walkers.sv.GqRecalibrator;

import org.apache.commons.math3.util.FastMath;

import java.util.List;

class FilterSummary {
    final int minGq;
    final long numMendelian;
    final long numDiscoverableTrios;
    final long numPassed;
    final long numVariants;
    final long numTruePositive;
    final long numFalsePositive;
    final long numFalseNegative;
    final long numTrueNegative;
    final boolean isLargeAlleleFrequency;
    final double truthWeight;
    final double inheritanceWeight;
    final String label;

    FilterSummary(final int minGq,
                  final long numMendelian, final long numDiscoverableTrios, final long numPassed, final long numVariants,
                  final long numTruePositive, final long numFalsePositive, final long numFalseNegative,
                  final long numTrueNegative, final boolean isLargeAlleleFrequency,
                  final double truthWeight, final double inheritanceWeight, final String label) {
        this.minGq = minGq;
        this.numMendelian = numMendelian;
        this.numDiscoverableTrios = numDiscoverableTrios;
        this.numPassed = numPassed;
        this.numVariants = numVariants;
        this.numTruePositive = numTruePositive;
        this.numFalsePositive = numFalsePositive;
        this.numFalseNegative = numFalseNegative;
        this.numTrueNegative = numTrueNegative;
        this.isLargeAlleleFrequency = isLargeAlleleFrequency;
        this.truthWeight = truthWeight;
        this.inheritanceWeight = inheritanceWeight;
        this.label = label;
        this.check();
    }

    FilterSummary(final FilterSummary other) {
        this.minGq = other.minGq;
        this.numMendelian = other.numMendelian;
        this.numDiscoverableTrios = other.numDiscoverableTrios;
        this.numPassed = other.numPassed;
        this.numVariants = other.numVariants;
        this.numTruePositive = other.numTruePositive;
        this.numFalsePositive = other.numFalsePositive;
        this.numFalseNegative = other.numFalseNegative;
        this.numTrueNegative = other.numTrueNegative;
        this.isLargeAlleleFrequency = other.isLargeAlleleFrequency;
        this.truthWeight = other.truthWeight;
        this.inheritanceWeight = other.inheritanceWeight;
        this.label = other.label;
    }

    static final FilterSummary EMPTY = new FilterSummary(
            Integer.MIN_VALUE, 0L, 0L, 0L, 0L,
            0L, 0L, 0L, 0L,
            false, 0.0, 0.0, null
    );

    void check() {
        if(numMendelian > numDiscoverableTrios && numDiscoverableTrios > 0) {
            throw new IllegalArgumentException("numMendelian (" + numMendelian + ") > numDiscoverable (" + numDiscoverableTrios + ")");
        }
        if(numPassed > numVariants) {
            throw new IllegalArgumentException("numPassedAlleleCount (" + numPassed + ") > maxPassedAlleleCount (" + numVariants + ")");
        }
        if(numMendelian < 0) {
            throw new IllegalArgumentException("numMendelian  (" + numMendelian + ") < 0");
        }
        if(numDiscoverableTrios < 0) {
            throw new IllegalArgumentException("numDiscoverable (" + numDiscoverableTrios + ") < 0");
        }
        if(numVariants < 0) {
            throw new IllegalArgumentException("numVariants (" + numVariants + ") < 0");
        }
        if(numPassed < 0) {
            throw new IllegalArgumentException("numPassed (" + numPassed + ") < 0");
        }

        if(numTruePositive < 0) {
            throw new IllegalArgumentException("numTruePositive  (" + numTruePositive + ") < 0");
        }
        if(numTrueNegative < 0) {
            throw new IllegalArgumentException("numTrueNegative  (" + numTrueNegative + ") < 0");
        }
        if(numFalseNegative < 0) {
            throw new IllegalArgumentException("numFalseNegative  (" + numFalseNegative + ") < 0");
        }
        if(numFalsePositive < 0) {
            throw new IllegalArgumentException("numFalsePositive  (" + numFalsePositive + ") < 0");
        }
    }

    boolean hasInheritanceData() { return this.numDiscoverableTrios > 0; }

    boolean hasOverlapData() {
        return this.numFalsePositive + this.numTruePositive + this.numFalseNegative + this.numTrueNegative > 0;
    }

    boolean isEmpty() {
        return !isNotEmpty();
    }

    boolean isNotEmpty() {
        return hasInheritanceData() || hasOverlapData();
    }

    FilterSummary setLabel(final String newLabel) {
        return new FilterSummary(minGq, numMendelian, numDiscoverableTrios, numPassed, numVariants,
                numTruePositive, numFalsePositive, numFalseNegative, numTrueNegative, isLargeAlleleFrequency,
                truthWeight, inheritanceWeight, newLabel);
    }

    FilterSummary add(final FilterSummary other) {
        return new FilterSummary(
            minGq,
            numMendelian + other.numMendelian,
            numDiscoverableTrios + other.numDiscoverableTrios,
            numPassed + other.numPassed,
            numVariants + other.numVariants,
            numTruePositive + other.numTruePositive,
            numFalsePositive + other.numFalsePositive,
            numFalseNegative + other.numFalseNegative,
            numTrueNegative + other.numTrueNegative,
            isLargeAlleleFrequency || other.isLargeAlleleFrequency,
            FastMath.max(truthWeight, other.truthWeight),
            FastMath.max(inheritanceWeight, other.inheritanceWeight),
            label == null ? other.label : label
        );
    }

    FilterSummary subtract(final FilterSummary other) {
        return new FilterSummary(
            minGq,
            numMendelian - other.numMendelian,
            numDiscoverableTrios - other.numDiscoverableTrios,
            numPassed - other.numPassed,
            numVariants - other.numVariants,
            numTruePositive - other.numTruePositive,
            numFalsePositive - other.numFalsePositive,
            numFalseNegative - other.numFalseNegative,
            numTrueNegative - other.numTrueNegative,
            isLargeAlleleFrequency,
            FastMath.max(truthWeight, other.truthWeight),
            FastMath.max(inheritanceWeight, other.inheritanceWeight),
            label == null ? other.label : label
        );
    }

    public static String tableHeader(final List<String> labelHeader) {
        return String.join("\t", labelHeader) + "\tnMendel\tnTrio\tnPassed\tnVar\tnTrue+\tnFalse+\tnTrue-\tnFalse-";
    }

    public String toTableLine() {
        return label + '\t' + numMendelian + '\t' + numDiscoverableTrios + '\t' + numPassed + '\t' + numVariants + '\t'
            + numTruePositive + '\t' + numFalsePositive + '\t' + numTrueNegative + '\t' + numFalsePositive;
    }

    @Override
    public String toString() {
        return label + ":{minGq:" + minGq + ",  numMendelian/Discoverable Trios:" + numMendelian + "/" + numDiscoverableTrios
                + ", numPassed:" + numPassed + "/" + numVariants
                + ", numTruePositive:" + numTruePositive
                + ", numFalsePositive:" + numFalsePositive + ", numFalseNegative: " + numFalseNegative
                + ", numTrueNegative:" + numTrueNegative + ", largeAlleleFrequency:" + isLargeAlleleFrequency + "}";
    }
}
