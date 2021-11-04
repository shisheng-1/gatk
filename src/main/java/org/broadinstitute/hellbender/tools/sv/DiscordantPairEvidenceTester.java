package org.broadinstitute.hellbender.tools.sv;

import com.google.common.collect.Sets;
import htsjdk.samtools.SAMSequenceDictionary;
import org.broadinstitute.hellbender.utils.Utils;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class DiscordantPairEvidenceTester {

    private final Map<String,Double> sampleCoverageMap;
    private final SAMSequenceDictionary dictionary;
    private final double backgroundRate;

    public DiscordantPairEvidenceTester(final Map<String, Double> sampleCoverageMap,
                                        final SAMSequenceDictionary dictionary,
                                        final double backgroundRate) {
        this.sampleCoverageMap = Utils.nonNull(sampleCoverageMap);
        this.dictionary = Utils.nonNull(dictionary);
        this.backgroundRate = backgroundRate;
    }

    public Double poissonTestRecord(final SVCallRecord record, final Collection<DiscordantPairEvidence> evidence) {
        Utils.nonNull(record);
        SVCallRecordUtils.validateCoordinatesWithDictionary(record, dictionary);

        // Depth-only calls cannot be refined
        if (record.isDepthOnly()) {
            return null;
        }

        final Set<String> callSamples = record.getAllSamples();
        final Set<String> calledSamples = record.getCarrierSampleSet();
        final Set<String> backgroundSamples = Sets.difference(callSamples, calledSamples);
        final int representativeDepth = EvidenceStatUtils.computeRepresentativeDepth(sampleCoverageMap.values());
        return poissonTest(evidence, calledSamples, backgroundSamples, representativeDepth);
    }

    public Double poissonTest(final Collection<DiscordantPairEvidence> evidence,
                              final Collection<String> carrierSamples,
                              final Collection<String> backgroundSamples,
                              final int representativeDepth) {
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(carrierSamples),
                "One or more carrier samples not found in sample coverage map");
        Utils.validateArg(sampleCoverageMap.keySet().containsAll(backgroundSamples),
                "One or more non-carrier samples not found in sample coverage map");

        // Default case
        if (evidence.isEmpty() || carrierSamples.isEmpty()) {
            return 1.;
        }
        final Map<String, Integer> sampleCounts = evidence.stream()
                .collect(Collectors.groupingBy(DiscordantPairEvidence::getSample,
                        Collectors.collectingAndThen(Collectors.toList(), List::size)));
        return EvidenceStatUtils.calculateOneSamplePoissonTest(sampleCounts, carrierSamples, backgroundSamples,
                sampleCoverageMap, representativeDepth, backgroundRate);
    }
}
