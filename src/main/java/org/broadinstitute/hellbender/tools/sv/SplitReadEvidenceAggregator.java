package org.broadinstitute.hellbender.tools.sv;

import com.google.common.collect.Ordering;
import htsjdk.samtools.SAMSequenceDictionary;
import org.broadinstitute.hellbender.engine.FeatureDataSource;
import org.broadinstitute.hellbender.utils.IntervalUtils;
import org.broadinstitute.hellbender.utils.SimpleInterval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SplitReadEvidenceAggregator extends SVEvidenceAggregator<SplitReadEvidence> {

    private final int window;
    private final boolean isStart; // Retrieve start position split reads, else end position

    public SplitReadEvidenceAggregator(final FeatureDataSource<SplitReadEvidence> source,
                                       final SAMSequenceDictionary dictionary,
                                       final int window,
                                       final boolean isStart) {
        super(source, dictionary, isStart ? "StartSplitReadSites" : "EndSplitReadSites");
        this.window = window;
        this.isStart = isStart;
    }

    public int getWindow() {
        return window;
    }

    @Override
    public SimpleInterval getEvidenceQueryInterval(final SVCallRecord call) {
        return (isStart ? call.getPositionAInterval() : call.getPositionBInterval()).expandWithinContig(window, dictionary);
    }

    @Override
    public boolean evidenceFilter(final SVCallRecord record, final SplitReadEvidence evidence) {
        if (isStart) {
            return evidence.getStrand() == record.getStrandA();
        } else {
            return evidence.getStrand() == record.getStrandB();
        }
    }

    public List<SplitReadSite> computeSites(final List<SplitReadEvidence> evidenceList) {
        if (!Ordering.from(IntervalUtils.getDictionaryOrderComparator(dictionary)).isOrdered(evidenceList)) {
            throw new IllegalArgumentException("Evidence list is not dictionary sorted");
        }
        final ArrayList<SplitReadSite> sites = new ArrayList<>();
        int position = 0;
        Map<String,Integer> sampleCounts = new HashMap<>();
        for (final SplitReadEvidence e : evidenceList) {
            if (e.getStart() != position) {
                if (!sampleCounts.isEmpty()) {
                    sites.add(new SplitReadSite(position, sampleCounts));
                    sampleCounts = new HashMap<>();
                }
                position = e.getStart();
            }
            if (e.getCount() > 0) {
                final String sample = e.getSample();
                sampleCounts.put(sample, e.getCount());
            }
        }
        if (!sampleCounts.isEmpty()) {
            sites.add(new SplitReadSite(position, sampleCounts));
        }
        sites.trimToSize();
        return sites;
    }
}
