package org.broadinstitute.hellbender.tools.sv;

import org.broadinstitute.hellbender.utils.Utils;

import java.util.Map;

/**
 * Container class for split read counts for multiple samples at a specific position
 */
final class SplitReadSite {
    private final int position;
    private final Map<String,Integer> sampleCountsMap;
    private final Double p;

    /**
     * @param position breakpoint position indicated by the split reads
     * @param sampleCountsMap map with (sample id, split read count > 0) entries
     */
    public SplitReadSite(final int position, final Map<String,Integer> sampleCountsMap, final Double p) {
        Utils.nonNull(sampleCountsMap);
        this.position = position;
        this.sampleCountsMap = sampleCountsMap;
        this.p = p;
    }

    public int getPosition() {
        return position;
    }

    public Double getP() {
        return p;
    }

    public int getCount(final String sample) {
        if (sampleCountsMap.containsKey(sample)) {
            return sampleCountsMap.get(sample);
        }
        return 0;
    }
}
