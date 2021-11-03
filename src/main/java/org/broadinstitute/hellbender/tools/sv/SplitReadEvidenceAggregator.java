package org.broadinstitute.hellbender.tools.sv;

import htsjdk.samtools.SAMSequenceDictionary;
import org.broadinstitute.hellbender.engine.FeatureDataSource;
import org.broadinstitute.hellbender.utils.SimpleInterval;

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
}
