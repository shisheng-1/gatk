package org.broadinstitute.hellbender.tools.sv;

import htsjdk.samtools.SAMSequenceDictionary;
import htsjdk.samtools.util.OverlapDetector;
import htsjdk.tribble.Feature;
import org.broadinstitute.hellbender.engine.FeatureDataSource;
import org.broadinstitute.hellbender.engine.ProgressMeter;
import org.broadinstitute.hellbender.utils.*;

import java.util.*;

public abstract class SVEvidenceAggregator<T extends Feature> {

    private final FeatureDataSource<T> source;
    private SimpleInterval cacheInterval;
    private Deque<T> cacheEvidence;
    private OverlapDetector<SimpleInterval> regionIntervalTree;
    private final ProgressMeter progressMeter;
    protected final SAMSequenceDictionary dictionary;

    private long numCacheHits;
    private long numCacheMisses;

    private static final long RECORDS_BETWEEN_TIME_CHECKS = 100L;

    public SVEvidenceAggregator(final FeatureDataSource<T> source,
                                final SAMSequenceDictionary dictionary,
                                final String progressLabel) {
        this.source = source;
        this.dictionary = dictionary;
        this.cacheInterval = null;
        this.cacheEvidence = null;
        this.progressMeter = new ProgressMeter();
        progressMeter.setRecordLabel(progressLabel);
        progressMeter.setRecordsBetweenTimeChecks(RECORDS_BETWEEN_TIME_CHECKS);
        numCacheHits = 0;
        numCacheMisses = 0;
    }

    public void addRegionIntervals(final Collection<SimpleInterval> intervals) {
        regionIntervalTree = OverlapDetector.create(mergeIntervals(intervals));
    }

    private List<SimpleInterval> mergeIntervals(final Collection<SimpleInterval> intervals) {
        final GenomeLocParser parser = new GenomeLocParser(dictionary);
        final List<GenomeLoc> rawLocs = IntervalUtils.genomeLocsFromLocatables(parser, intervals);
        final GenomeLocSortedSet mergedLocs = IntervalUtils.sortAndMergeIntervals(parser, rawLocs, IntervalMergingRule.ALL);
        return IntervalUtils.convertGenomeLocsToSimpleIntervals(mergedLocs.toList());
    }

    abstract public SimpleInterval getEvidenceQueryInterval(final SVCallRecord record);
    public boolean evidenceFilter(final SVCallRecord record, final T evidence) { return true; }

    private SimpleInterval getRegionInterval(final SimpleInterval interval) {
        final Set<SimpleInterval> evidenceIntervals = regionIntervalTree.getOverlaps(interval);
        Utils.validate(evidenceIntervals.size() == 1, "Expected exactly 1 evidence interval but " +
                "found " + evidenceIntervals.size());
        return evidenceIntervals.iterator().next();
    }

    public List<T> collectEvidence(final SVCallRecord call) {
        Utils.nonNull(call);

        final SimpleInterval callInterval = getEvidenceQueryInterval(call);
        if (regionIntervalTree == null) {
            return source.queryAndPrefetch(callInterval);
        }
        final SimpleInterval regionInterval = getRegionInterval(callInterval);
        if (!regionInterval.equals(cacheInterval)) {
            numCacheMisses++;
            cacheEvidence = new ArrayDeque<>(source.queryAndPrefetch(regionInterval));
            cacheInterval = regionInterval;
        } else {
            numCacheHits++;
        }
        while (!cacheEvidence.isEmpty() && callInterval.getStart() > cacheEvidence.peek().getStart()) {
            cacheEvidence.pop();
        }
        final List<T> callEvidence = new ArrayList<>();
        for (final T evidence : cacheEvidence) {
            if (callInterval.overlaps(evidence)) {
                if (evidenceFilter(call, evidence)) {
                    callEvidence.add(evidence);
                }
            } else {
                break;
            }
        }

        if (progressMeter.started()) {
            progressMeter.update(callInterval);
        }
        return callEvidence;
    }

    public double getAndResetCacheHitRate() {
        final double rate = numCacheHits / (double) (numCacheMisses + numCacheHits);
        numCacheHits = 0;
        numCacheMisses = 0;
        return rate;
    }
}
