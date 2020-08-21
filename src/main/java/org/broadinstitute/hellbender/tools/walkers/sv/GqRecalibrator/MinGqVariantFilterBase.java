package org.broadinstitute.hellbender.tools.walkers.sv.GqRecalibrator;

import htsjdk.samtools.util.Locatable;
import htsjdk.variant.variantcontext.*;
import htsjdk.variant.variantcontext.writer.Options;
import htsjdk.variant.variantcontext.writer.VariantContextWriter;
import htsjdk.variant.vcf.*;
import net.minidev.json.JSONArray;
import net.minidev.json.JSONObject;
import net.minidev.json.JSONValue;
import net.minidev.json.parser.ParseException;
import org.broadinstitute.barclay.argparser.Argument;
import org.broadinstitute.hellbender.cmdline.StandardArgumentDefinitions;
import org.broadinstitute.hellbender.engine.*;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.exceptions.UserException;
import org.broadinstitute.hellbender.utils.SimpleInterval;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.samples.PedigreeValidationType;
import org.broadinstitute.hellbender.utils.samples.SampleDBBuilder;
import org.broadinstitute.hellbender.utils.samples.Trio;

import java.io.*;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.*;

import org.apache.commons.math3.util.FastMath;
import org.broadinstitute.hellbender.utils.variant.GATKVariantContextUtils;
import org.jetbrains.annotations.NotNull;

import static org.apache.commons.math3.util.FastMath.*;


/**
 * Extract matrix of properties for each variant.
 * Also extract, numVariants x numTrios x 3 tensors of allele count and genotype quality.
 * These data will be used to train a variant filter based on min GQ" (and stratified by other variant properties) that
 * maximizes the admission of variants with Mendelian inheritance pattern while omitting non-Mendelian variants.
 *
 * Derived class must implement abstract method trainFilter()
 */
public abstract class MinGqVariantFilterBase extends VariantWalker {
    @Argument(fullName=StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME,
              shortName=StandardArgumentDefinitions.PEDIGREE_FILE_SHORT_NAME,
              doc="Pedigree file, necessary for \"Train\" mode, ignored in \"Filter\" mode.", optional=true)
    public GATKPath pedigreeFile = null;

    @Argument(fullName=StandardArgumentDefinitions.OUTPUT_LONG_NAME,
              shortName=StandardArgumentDefinitions.OUTPUT_SHORT_NAME,
              doc="Output VCF produced in \"Filter\" mode, ignored in \"Train\" mode.", optional=true)
    public GATKPath outputFile = null;

    @Argument(fullName="model-file", shortName="m",
              doc="Path to saved pre-existing filter model. In \"Filter\" mode, this is a mandatory argument."
                 +" In \"Train\" mode, this is an optional argument for additional training based on an existing model.")
    public GATKPath modelFile = null;

    @Argument(fullName="properties-table-file", optional=true,
            doc="Path to save properties table as JSON for analysis outside of this program. Only has an effect in TRAIN mode.")
    public GATKPath propertiesTableFile = null;

    @Argument(fullName="truth-file", shortName="t", optional=true,
              doc="Path to JSON file with truth data. Keys are sample IDs and values are objects with key \"good\""
                 +" corresponding to a list of known true variant IDs, and key \"bad\" corresponding to a list of known"
                 +" bad variant IDs")
    public GATKPath truthFile = null;

    private enum RunMode {Train, Filter}
    @Argument(fullName="mode", doc="Mode of operation: either \"Train\" or \"Filter\"")
    public RunMode runMode;

    public enum TrainObjective {PerVariantOptimized, LogLoss}
    @Argument(fullName="objective", optional=true,
             doc="Choose clasifier training objective: \n"
                + "\tPerVariantOptimized: match truth values obtained via per-variant optimal min-GQ\n"
                + "\tLogLoss: optimize log-loss of pMendelian and pOverlap directly")
    public TrainObjective trainObjective = TrainObjective.PerVariantOptimized;

    @Argument(fullName="keep-homvar", shortName="kh", doc="Keep homvar variants even if their GQ is less than min-GQ", optional=true)
    public boolean keepHomvar = true;

    @Argument(fullName="keep-multiallelic", shortName="km", doc="Keep multiallelic variants even if their GQ is less than min-GQ", optional=true)
    public boolean keepMultiallelic = true;

    @Argument(fullName="validation-proportion", shortName="vp", doc="Proportion of variants to set aside for cross-validation",
              optional=true, minValue=0.0, maxValue=1.0)
    public double validationProportion = 0.2;

    @Argument(fullName="report-min-gq-filter-threshold", shortName="rf", optional=true, minValue=0.0, maxValue=1.0,
              doc="Add \"" + EXCESSIVE_MIN_GQ_FILTER_KEY + "\" to FILTER if the proportion of samples with calls filtered by "
                 + MIN_GQ_KEY + " is greater than this threshold.")
    public double reportMinGqFilterThreshold = 0.005;

    @Argument(fullName="truth-weight", shortName="tw", optional=true, minValue=0.0, maxValue=1.0,
            doc="Weight for truth data in loss function.")
    public static double truthWeight = 0.5;

    @Argument(fullName="print-progress", shortName="p", doc="Print progress of fit during training", optional = true)
    public int printProgress = 1;

    static final String minSamplesToEstimateAlleleFrequencyKey = "min-samples-to-estimate-allele-frequency";
    @Argument(fullName=minSamplesToEstimateAlleleFrequencyKey, shortName="ms", optional=true,
              doc="If the VCF does not have allele frequency, estimate it from the sample population if there are at least this many samples. Otherwise throw an exception.")
    public int minSamplesToEstimateAlleleFrequency = 100;

    @Argument(fullName="genome-tract", shortName="gt", optional=true)
    final List<String> genomeTractFiles = new ArrayList<>();

    @Argument(fullName="minGQ", shortName="gq", optional=true)
    int minGQ = p_to_scaled_logits(0.5);

    @Argument(fullName="prediction-scale-factor", doc="Scale factor for raw predictions to logits", optional=true)
    public double predictionScaleFactor = 1.0;
    final double phredCoef = FastMath.log(10.0) / 10.0;

    @Argument(fullName="target-precision", doc="Desired minimum precision to achieve before maximizing recall", optional=true)
    public static double targetPrecision = 0.90;

    @Argument(fullName="max-inheritance-af", optional=true,
        doc="Max allele frequency where inheritance will be considered truthful. AF in range (mIAf, 1.0 - mIAf) have no score"
    )
    public static double maxInheritanceAf = 0.2;

    @Argument(fullName="strict-mendelian", optional=true,
            doc="If true, apply normal rules for mendelian inheritance; if false, allow confusion between HET and HOMVAR"
    )
    public static boolean strictMendelian = true;

    List<TractOverlapDetector> tractOverlapDetectors = null;

    static final Map<String, double[]> propertyBinsMap = new LinkedHashMap<>();
    // numVariants array of property-category bins
    protected int[] propertyBins = null;
    int numPropertyBins;
    protected long[] variantsPerPropertyBin = null;
    protected String[] propertyBinLabels = null;
    protected List<String> propertyBinLabelColumns = null;
    protected double[] propertyBinInheritanceWeights = null;
    protected double[] propertyBinTruthWeights = null;

    protected List<String> sampleIds = null; // numSamples list of IDs for samples that will be used for training
    protected int[][] trioSampleIndices = null; // numTrios x 3 matrix of sample indices (paternal, maternal, child) for each trio
    protected Set<Integer> allTrioSampleIndices; // set of all sample indices that are in a trio
    protected final Map<String, Integer> sampleIndices = new HashMap<>(); // map from sampleId to numerical index, for samples used in training
    protected final PropertiesTable propertiesTable = new PropertiesTable();
    protected List<String> variantIds = new ArrayList<>();

    private static final int ALLELE_COUNT_BITMASK = 3;
    private static final int NO_CALL_BIT_SHIFT = 2;
    // numVariants x numSamples matrix of allele counts, packed to efficiently hold called allele counts and no-calls:
    private final List<byte[]> packedSampleVariantCounts = new ArrayList<>();
    // numVariants x numSamples matrix of genotype qualities:
    protected List<int[]> sampleVariantGenotypeQualities = new ArrayList<>();
    // map from variantIndex to array of known good GQ values for that variant
    protected Map<Integer, int[]> goodVariantGqs = null;
    // map from variantIndex to array of known good/bad sample indices for that variant
    protected final Map<Integer, Set<Integer>> goodSampleVariantIndices = new HashMap<>();
    protected final Map<Integer, Set<Integer>> badSampleVariantIndices = new HashMap<>();
    // map from variantIndex to array of known bad GQ values for that variant
    protected Map<Integer, int[]> badVariantGqs = null;

    protected Random randomGenerator = Utils.getRandomGenerator();

    private VariantContextWriter vcfWriter = null;

    private int numVariants;
    private int numSamples;
    private int numTrios;

    private static final float PROB_EPS = 1.0e-3F;
    private static final Set<String> BREAKPOINT_SV_TYPES = new HashSet<>(Arrays.asList("BND", "CTX"));
    private static final double SV_EXPAND_RATIO = 1.0; // ratio of how much to expand SV gain range to SVLEN
    private static final int BREAKPOINT_HALF_WIDTH = 50; // how wide to make breakpoint intervals for tract overlap
    private static final String[] DISTAL_TARGETS_KEYS = {"SOURCE", "CPX_INTERVALS"}; // keys for distal targets in VCF entries
    private static final String SVLEN_KEY = "SVLEN";
    private static final String EVIDENCE_KEY = "EVIDENCE";
    private static final String FILTER_KEY = "FILTER";
    private static final String NO_EVIDENCE = "NO_EVIDENCE";
    private static final String ALGORITHMS_KEY = "ALGORITHMS";
    private static final String NO_ALGORITHM = "NO_ALGORITHM";
    private static final String MIN_GQ_KEY = "MINGQ";
    private static final String EXCESSIVE_MIN_GQ_FILTER_KEY = "LOW_QUALITY";
    private static final String MULTIALLELIC_FILTER = "MULTIALLELIC";
    private static final String GOOD_VARIANT_TRUTH_KEY = "good_variant_ids";
    private static final String BAD_VARIANT_TRUTH_KEY = "bad_variant_ids";
    private static final String CHR2_KEY = "CHR2";

    protected static final int FATHER_IND = 0;
    protected static final int MOTHER_IND = 1;
    protected static final int CHILD_IND = 2;

    private static final String PE_GQ_KEY = "PE_GQ";
    private static final String RD_GQ_KEY = "RD_GQ";
    private static final String SR_GQ_KEY = "SR_GQ";
    private static final int MISSING_GQ_VAL = -1;

    // properties used to gather main matrix / tensors during apply()
    private Set<Trio> pedTrios = null;
    private Map<String, Set<String>> goodSampleVariants = null;
    private Map<String, Set<String>> badSampleVariants = null;


    // train/validation split indices
    private int[] trainingIndices;
    private int[] validationIndices;

    // properties for calculating f1 score or estimating its pseudo-derivatives
    private int[] perVariantOptimalMinGq = null; // <- REMOVE
    protected int[] maxDiscoverableMendelianTrioCount = null;
    long numDiscoverableMendelianTrios = 0;
    protected double optimalProportionOfSampleVariantsPassing;

    protected final int getNumVariants() { return propertiesTable.getNumRows(); }
    protected final int getNumSamples() { return numSamples; }
    protected final int getNumTrios() { return numTrios; }
    protected final int getNumProperties() { return propertiesTable.getNumProperties(); }
    protected final int[] getTrainingIndices() { return trainingIndices; }
    protected final int[] getValidationIndices() { return validationIndices; }

    // stats on tool actions, input/output VCFs
    private int numFilterableGenotypes;
    private int numFilteredGenotypes;
    private int numInputVar;
    private int numInputRef;
    private int numInputNoCall;
    private int numOutputVar;
    private int numOutputRef;
    private int numOutputNoCall;

    /**
     * Entry-point function to initialize the samples database from input data
     */
    private void getPedTrios() {
        if(pedigreeFile == null) {
            throw new UserException.BadInput(StandardArgumentDefinitions.PEDIGREE_FILE_LONG_NAME
                                             + " must be specified in \"TRAIN\" mode");
        }
        final SampleDBBuilder sampleDBBuilder = new SampleDBBuilder(PedigreeValidationType.STRICT);
        sampleDBBuilder.addSamplesFromPedigreeFiles(Collections.singletonList(pedigreeFile));

        final Set<String> vcfSamples = new HashSet<>(getHeaderForVariants().getGenotypeSamples());
        // get the trios from the pedigree file, keeping only those that are fully present in the input VCF
        pedTrios = sampleDBBuilder.getFinalSampleDB()
            .getTrios()
            .stream()
            .filter(trio -> vcfSamples.contains(trio.getPaternalID()) && vcfSamples.contains(trio.getMaternalID())
                            && vcfSamples.contains(trio.getChildID()))
            .collect(Collectors.toSet());
        if(pedTrios.isEmpty()) {
            throw new UserException.BadInput(
                "The pedigree file (" + pedigreeFile + ") does not contain any trios that are present in the input VCF "
                + "(" + drivingVariantFile + ")"
            );
        }
        numTrios = pedTrios.size();
        // collect ped trios into training samples
        pedTrios.stream()
                .flatMap(trio -> Stream.of(trio.getPaternalID(), trio.getMaternalID(), trio.getChildID()))
                .forEach(this::addSampleIndex);
        // keep track of the samples that are in trios (those will always have "truth" through inheritance)
        trioSampleIndices = pedTrios.stream()
                .map(
                    trio -> Stream.of(trio.getPaternalID(), trio.getMaternalID(), trio.getChildID())
                            .mapToInt(sampleIndices::get)
                            .toArray()
                )
                .toArray(int[][]::new);

        allTrioSampleIndices = pedTrios.stream()
                .flatMap(trio -> Stream.of(trio.getPaternalID(), trio.getMaternalID(), trio.getChildID()))
                .map(sampleIndices::get)
                .collect(Collectors.toSet());

    }

    private void loadVariantTruthData() {
        if(truthFile == null) {
            System.out.println("No truth file specified. Not using truth data.");
            return;
        } else {
            System.out.println("Loading truth data from " + truthFile);
        }
        final JSONObject jsonObject;
        try (final InputStream fileInputStream = truthFile.getInputStream()){
            jsonObject = (JSONObject) JSONValue.parseWithException(fileInputStream);
        } catch (IOException | ParseException ioException) {
            throw new GATKException("Unable to parse JSON from inputStream", ioException);
        }
        final Set<String> vcfSamples = new HashSet<>(getHeaderForVariants().getGenotypeSamples());
        goodSampleVariants = new HashMap<>();
        badSampleVariants = new HashMap<>();
        for(final Map.Entry<String, Object> sampleTruthEntry : jsonObject.entrySet()) {
            final String sampleId = sampleTruthEntry.getKey();
            if(!vcfSamples.contains(sampleId)) {
                continue; // don't bother storing truth data that doesn't overlap the input VCF
            }
            final JSONObject sampleTruth = (JSONObject)sampleTruthEntry.getValue();
            for(final Object variantIdObj : (JSONArray)sampleTruth.get(GOOD_VARIANT_TRUTH_KEY)) {
                final String variantId = (String)variantIdObj;
                if(goodSampleVariants.containsKey(variantId)) {
                    goodSampleVariants.get(variantId).add(sampleId);
                } else {
                    goodSampleVariants.put(variantId, new HashSet<>(Collections.singleton(sampleId)) );
                }
            }
            for(final Object variantIdObj : (JSONArray)sampleTruth.get(BAD_VARIANT_TRUTH_KEY)) {
                final String variantId = (String)variantIdObj;
                if(badSampleVariants.containsKey(variantId)) {
                    badSampleVariants.get(variantId).add(sampleId);
                } else {
                    badSampleVariants.put(variantId, new HashSet<>(Collections.singleton(sampleId)));
                }
            }
        }
        if(goodSampleVariants.isEmpty() && badSampleVariants.isEmpty()) {
            System.out.println("Truth file specified (" + truthFile + "), but no samples/variants overlap with input VCF ("
                               + drivingVariantFile + "). Not using truth data.");
            goodSampleVariants = null;
            badSampleVariants = null;
            return;
        }
        // Add any new samples that have truth but are not in trios file to the training samples
        goodSampleVariants.values().stream().flatMap(Collection::stream).forEach(this::addSampleIndex);
        badSampleVariants.values().stream().flatMap(Collection::stream).forEach(this::addSampleIndex);
        // prepare to hold data for scoring
        goodVariantGqs = new HashMap<>();
        badVariantGqs = new HashMap<>();
    }

    private void addSampleIndex(final String sampleId) {
        if(!sampleIndices.containsKey(sampleId)) {
            sampleIndices.put(sampleId, sampleIndices.size());
        }
    }

    private void setSampleIds() {
        sampleIds = sampleIndices.entrySet()
            .stream()
            .sorted(Comparator.comparingInt(Map.Entry::getValue))
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
        numSamples = sampleIndices.size();
    }

    private void initializeVcfWriter() {
        List<Options> options = new ArrayList<>();
        //if (createOutputVariantIndex) {
            options.add(Options.INDEX_ON_THE_FLY);
        //}
        vcfWriter = GATKVariantContextUtils.createVCFWriter(
                outputFile.toPath(),
                getHeaderForVariants().getSequenceDictionary(),
                createOutputVariantMD5,
                options.toArray(new Options[0]));
        //vcfWriter = createVCFWriter(outputFile);
        final Set<VCFHeaderLine> hInfo = new LinkedHashSet<>(getHeaderForVariants().getMetaDataInInputOrder());
        final String filterableVariant = (keepMultiallelic ? "biallelic " : "") + (keepHomvar ? "het " : "") + "variant";
        hInfo.add(new VCFInfoHeaderLine(MIN_GQ_KEY, 1, VCFHeaderLineType.Integer,
                            "Minimum passing GQ for each " + filterableVariant));
        hInfo.add(new VCFFilterHeaderLine(EXCESSIVE_MIN_GQ_FILTER_KEY,
                               "More than " + (100 * reportMinGqFilterThreshold) + "% of sample GTs were masked as no-call GTs due to low GQ"));
        final VCFHeader vcfHeader = new VCFHeader(hInfo, getHeaderForVariants().getGenotypeSamples());
        vcfWriter.setHeader(vcfHeader);
        vcfWriter.writeHeader(vcfHeader);
    }

    @Override
    public void onTraversalStart() {
        // set propertyBinsMap bins
        propertyBinsMap.put(SVLEN_KEY, new double[] {50.0, 500.0, 5000.0});
        propertyBinsMap.put(VCFConstants.ALLELE_FREQUENCY_KEY, new double[] {maxInheritanceAf, 1.0 - maxInheritanceAf});

        loadTrainedModel();  // load model and saved properties stats
        tractOverlapDetectors = genomeTractFiles.stream()  // load genome tracts
            .map(genomeTractFile -> {
                    try {
                        return new TractOverlapDetector(genomeTractFile);
                    } catch(IOException ioException) {
                        throw new GATKException("Error loading " + genomeTractFile, ioException);
                    }
                })
            .collect(Collectors.toList());

        numInputVar = 0;
        numInputRef = 0;
        numInputNoCall = 0;
        if(runMode == RunMode.Train) {
            getPedTrios();  // get trios from pedigree file
            loadVariantTruthData(); // load variant truth data from JSON file
            // at this point, included samples will only be those that are in a trio or have some variant truth data
            setSampleIds(); // set data organizing training samples
        } else {
            getHeaderForVariants().getGenotypeSamples().forEach(this::addSampleIndex); // use all samples in VCF
            setSampleIds();
            initializeVcfWriter();  // initialize vcfWriter and write header
            numFilterableGenotypes = 0;
            numFilteredGenotypes = 0;
            numOutputVar = 0;
            numOutputRef = 0;
            numOutputNoCall = 0;
            propertiesTable.setNumAllocatedRows(1);
        }
    }

    private static boolean mapContainsTrio(final Map<String, Integer> map, final Trio trio) {
        return map.containsKey(trio.getPaternalID()) && map.containsKey(trio.getMaternalID())
                && map.containsKey(trio.getChildID());
    }

    private static int[] getMappedTrioProperties(final Map<String, Integer> map, final Trio trio) {
        return new int[] {map.get(trio.getPaternalID()), map.get(trio.getMaternalID()), map.get(trio.getChildID())};
    }

    private boolean getIsMultiallelic(final VariantContext variantContext) {
        return variantContext.getNAlleles() > 2 || variantContext.getFilters().contains(MULTIALLELIC_FILTER);
    }

    private boolean getVariantIsFilterable(final VariantContext variantContext, final Map<String, Integer> sampleAlleleCounts) {
        final boolean maybeFilterable = !(keepMultiallelic && getIsMultiallelic(variantContext));
        if(maybeFilterable) {
            if(runMode == RunMode.Filter) {
                // filter no matter what, because end-user may be interested in minGQ
                return true;
            } else {
                // check if any of the allele counts can be filtered so as not to train on unfilterable variants
                return pedTrios.stream()
                        .filter(trio -> mapContainsTrio(sampleAlleleCounts, trio))
                        .flatMapToInt(trio -> Arrays.stream(getMappedTrioProperties(sampleAlleleCounts, trio)))
                        .anyMatch(keepHomvar ? ac -> ac == 1 : ac -> ac > 0);
            }
        } else {
            return false;
        }
    }

    private byte[] packSampleAlleleCounts(final int[] sampleAlleleCounts, final int[] sampleNoCallCounts) {
        final byte[] packedSampleAlleleCounts = new byte[sampleAlleleCounts.length];
        for(int idx = 0; idx < sampleAlleleCounts.length; ++idx) {
            packedSampleAlleleCounts[idx] = (byte)(
                sampleAlleleCounts[idx] + (sampleNoCallCounts[idx] << NO_CALL_BIT_SHIFT)
            );
        }
        if(!Arrays.equals(sampleAlleleCounts, unpackAlleleCounts(packedSampleAlleleCounts))) {
            throw new IllegalArgumentException("Unpacked sampleAlleleCounts don't agree with original");
        }
        if(!Arrays.equals(sampleNoCallCounts, unpackNoCallCounts(packedSampleAlleleCounts))) {
            throw new IllegalArgumentException("Unpacked sampleNoCallCounts don't agree with original");
        }
        return packedSampleAlleleCounts;
    }

    private static int[] unpackAlleleCounts(final byte[] packedSampleAlleleCounts) {
        return IntStream.range(0, packedSampleAlleleCounts.length).map(
                idx -> packedSampleAlleleCounts[idx] & ALLELE_COUNT_BITMASK
        ).toArray();
    }

    private static int[] unpackNoCallCounts(final byte[] packedSampleAlleleCounts) {
        return IntStream.range(0, packedSampleAlleleCounts.length).map(
                idx -> packedSampleAlleleCounts[idx] >> NO_CALL_BIT_SHIFT
        ).toArray();
    }

    private int[] getSampleAlleleCounts(final int variantIndex) {
        final byte[] packedSampleAlleleCounts = packedSampleVariantCounts.get(variantIndex);
//        return IntStream.range(0, packedSampleAlleleCounts.length).map(
//            idx -> packedSampleAlleleCounts[idx] & ALLELE_COUNT_BITMASK
//        ).toArray();
        return unpackAlleleCounts(packedSampleAlleleCounts);
    }

    private int getSampleVariantAlleleCounts(final int variantIndex, final int sampleIndex) {
        return packedSampleVariantCounts.get(variantIndex)[sampleIndex] & ALLELE_COUNT_BITMASK;
    }

    private int[] getSampleNoCallCounts(final int variantIndex) {
        final byte[] packedSampleAlleleCounts = packedSampleVariantCounts.get(variantIndex);
//        return IntStream.range(0, packedSampleAlleleCounts.length).map(
//                idx -> packedSampleAlleleCounts[idx] >> NO_CALL_BIT_SHIFT
//        ).toArray();
        return unpackNoCallCounts(packedSampleAlleleCounts);
    }

    /**
     * form numTrios x 3 matrix of allele counts for specified variantIndex (paternal, maternal, child)
     */
    protected int[][] getTrioAlleleCountsMatrix(final int variantIndex) {
        final int[] sampleAlleleCounts = getSampleAlleleCounts(variantIndex);
        return Arrays.stream(trioSampleIndices)
            .map(
                trioIndices -> Arrays.stream(trioIndices)
                                .map(idx -> sampleAlleleCounts[idx])
                                .toArray()
            )
            .toArray(int[][]::new);
    }

    /**
     * form numTrios x 3 matrix of no-call counts for specified variantIndex (paternal, maternal, child)
     */
    protected int[][] getTrioNoCallCountsMatrix(final int variantIndex) {
        final int[] sampleNoCallCounts = getSampleNoCallCounts(variantIndex);
        return Arrays.stream(trioSampleIndices)
                .map(
                        trioIndices -> Arrays.stream(trioIndices)
                                .map(idx -> sampleNoCallCounts[idx])
                                .toArray()
                )
                .toArray(int[][]::new);
    }

    /**
     * form numTrios x 3 matrix of genotype qualities for specified variantIndex (paternal, maternal, child)
     */
    protected int[][] getTrioGenotypeQualitiesMatrix(final int variantIndex) {
        final int[] sampleGenotypeQualities = sampleVariantGenotypeQualities.get(variantIndex);
        return Arrays.stream(trioSampleIndices)
                .map(
                        trioIndices -> Arrays.stream(trioIndices)
                                .map(idx -> sampleGenotypeQualities[idx])
                                .toArray()
                )
                .toArray(int[][]::new);
    }


    protected boolean getSampleVariantIsFilterable(final int variantIndex, final int sampleIndex) {
        return alleleCountIsFilterable(getSampleVariantAlleleCounts(variantIndex, sampleIndex));
    }

    /**
     * A sample variant is filterable if it's het, or if it's HOMVAR and keepHomVar is not true
     */
    protected boolean alleleCountIsFilterable(final int alleleCount) {
        return keepHomvar ? alleleCount == 1 : alleleCount > 0;
    }

    /**
     * A sample Variant is trainable if a) it is filterable, and b) there is truth data (inheritance or known good/bad)
     */
    protected boolean getSampleVariantIsTrainable(final int variantIndex, final int sampleIndex) {
        if(allTrioSampleIndices.contains(sampleIndex) ||
           goodSampleVariantIndices.getOrDefault(variantIndex, Collections.emptySet()).contains(sampleIndex) ||
           badSampleVariantIndices.getOrDefault(variantIndex, Collections.emptySet()).contains(sampleIndex)) {
            return getSampleVariantIsFilterable(variantIndex, sampleIndex);
        } else {
            return false;
        }
    }

    private SimpleInterval getDistalTarget(final String targetString) {
        return new SimpleInterval(targetString.substring(targetString.indexOf("_") + 1));
    }

    private void getTractProperties(final TractOverlapDetector tractOverlapDetector,
                                    final VariantContext variantContext) {
        // get range of variant (use expanded location for insertions or duplications)
        final List<Locatable> overlapCheckLocations = new ArrayList<>();
        final String svType = variantContext.getAttributeAsString(VCFConstants.SVTYPE, null);
        if(BREAKPOINT_SV_TYPES.contains(svType)) {
            // Add checks at the breakpoint locations. They are points so we want to check tract overlaps in the regions
            // around them
            overlapCheckLocations.add(
                new SimpleInterval(variantContext.getContig(),
                             FastMath.max(1, variantContext.getStart() - BREAKPOINT_HALF_WIDTH),
                             variantContext.getStart() + BREAKPOINT_HALF_WIDTH)
            );
            overlapCheckLocations.add(
                new SimpleInterval(variantContext.getAttributeAsString(CHR2_KEY, null),
                             FastMath.max(1, variantContext.getEnd() - BREAKPOINT_HALF_WIDTH),
                              variantContext.getEnd() + BREAKPOINT_HALF_WIDTH)
            );
        } else {
            // Insertions can have zero width but they have uncertainty in their true location. Expand their central
            // interval for the purpose of checking tract overlap. Do this for any type that's "narrow"
            final int width = variantContext.getEnd() - variantContext.getStart();
            final boolean is_narrow = width < 2 * BREAKPOINT_HALF_WIDTH;
            final int expand = is_narrow ?
                max(
                    (int)(abs(variantContext.getAttributeAsInt(SVLEN_KEY, 0)) * SV_EXPAND_RATIO),
                    BREAKPOINT_HALF_WIDTH
                ) :
                0;
            overlapCheckLocations.add(
                new SimpleInterval(variantContext.getContig(),
                    FastMath.max(1, variantContext.getStart() - expand),
                    variantContext.getStart() + expand)
            );
            // if the interval isn't narrow, add the end points as potential breakpoints to check
            if(!is_narrow) {
                overlapCheckLocations.add(
                    new SimpleInterval(variantContext.getContig(),
                                      FastMath.max(1, variantContext.getStart() - BREAKPOINT_HALF_WIDTH),
                                      variantContext.getStart() + BREAKPOINT_HALF_WIDTH)
                );
                overlapCheckLocations.add(
                    new SimpleInterval(variantContext.getContig(),
                                 FastMath.max(1, variantContext.getEnd() - BREAKPOINT_HALF_WIDTH),
                                 variantContext.getEnd() + BREAKPOINT_HALF_WIDTH)
                );
            }
        }
        // Check for other distal locations in "SOURCE" field
        //   they may be null (filter out)
        //   they may be of form SVTYPE_SIMPLEINTERVAL (chop off everything before "_")
        Arrays.stream(DISTAL_TARGETS_KEYS)
                .map(key -> variantContext.getAttributeAsStringList(key, null))
                .filter(Objects::nonNull)
                .flatMap(List::stream)
                .map(this::getDistalTarget)
                .forEach(overlapCheckLocations::add);

        final double[] overlaps;
        if(tractOverlapDetector.hasOther()) { // This tract has paired intervals
            overlaps = overlapCheckLocations.stream()
                    .mapToDouble(location -> tractOverlapDetector.getPrimaryOverlapFraction(location)
                                             + tractOverlapDetector.getOtherOverlapfraction(location))
                    .toArray();
            // check for spanning by streaming over all possible pairs
            final boolean spans = IntStream.range(0, overlapCheckLocations.size() - 1)
                .anyMatch(
                    i -> IntStream.range(i + 1, overlapCheckLocations.size()).anyMatch(
                        j -> tractOverlapDetector.spansPrimaryAndOther(overlapCheckLocations.get(i),
                                                                       overlapCheckLocations.get(j))
                    )
                );
            propertiesTable.append(
                tractOverlapDetector.getName() + "_spans",
                spans
            );
        } else {  // this tract has simple intervals
            overlaps = overlapCheckLocations.stream()
                .mapToDouble(tractOverlapDetector::getPrimaryOverlapFraction)
                .toArray();
        }
        // breakpoint types have no "center", otherwise get overlap of primary location
        propertiesTable.append(
            tractOverlapDetector.getName() + "_center",
            BREAKPOINT_SV_TYPES.contains(svType) ? 0.0 : overlaps[0]
        );
        propertiesTable.append(
            tractOverlapDetector.getName() + "_min",
            Arrays.stream(overlaps).min().orElse(0.0)
        );
        propertiesTable.append(
            tractOverlapDetector.getName() + "_max",
            Arrays.stream(overlaps).max().orElse(0.0)
        );
    }

    public int getGenotypeAttributeAsInt(final Genotype genotype, final String key, Integer defaultValue) {
        Object x = genotype.getExtendedAttribute(key);
        if ( x == null || x == VCFConstants.MISSING_VALUE_v4 ) {
            if(defaultValue == null) {
                throw new IllegalArgumentException("Genotype is missing value of " + key);
            } else {
                return defaultValue;
            }
        }
        if ( x instanceof Integer ) return (Integer)x;
        return Integer.parseInt((String)x); // throws an exception if this isn't a string
    }


    private int[] getGenotypeAttributeAsInt(final Iterable<Genotype> sampleGenotypes, final String attributeKey,
                                            @SuppressWarnings("SameParameterValue")final int missingAttributeValue) {
        return StreamSupport.stream(sampleGenotypes.spliterator(), false)
                        .mapToInt(g -> getGenotypeAttributeAsInt(g, attributeKey, missingAttributeValue))
                        .toArray();
    }

    private final Map<String, Integer> applySampleAlleleCounts = new HashMap<>();  // allocate once to avoid thrashing memory
    private final Map<String, Integer> applySampleNoCallCounts = new HashMap<>();  // allocate once to avoid thrashing memory

    /**
     * Accumulate properties for variant matrix, and allele counts, genotype quality for trio tensors
     */
    @Override
    public void apply(VariantContext variantContext, ReadsContext readsContext, ReferenceContext ref, FeatureContext featureContext) {
        // get per-sample allele counts as a map indexed by sample ID
        int numCalledAlleles = 0;
        int numNonRefAlleles = 0;
        int numVariantInputVar = 0;
        int numVariantInputNoCall = 0;
        int numVariantInputRef = 0;
        for(final Genotype genotype : variantContext.getGenotypes()) {
            int noCallCounts = 0;
            int nonRefCounts = 0;
            for(final Allele allele : genotype.getAlleles()) {
                if(allele.isNoCall()) {
                    ++noCallCounts;
                } else if(!allele.isReference()) {
                    ++nonRefCounts;
                }
            }
            applySampleAlleleCounts.put(genotype.getSampleName(), nonRefCounts);
            applySampleNoCallCounts.put(genotype.getSampleName(), noCallCounts);
            if(nonRefCounts > 0) {
                ++numVariantInputVar;
            } else if(noCallCounts > 0) {
                ++numVariantInputNoCall;
            } else {
                ++numVariantInputRef;
            }
            numNonRefAlleles += nonRefCounts;
            numCalledAlleles += genotype.getAlleles().size() - noCallCounts;
        }
        numInputVar += numVariantInputVar;
        numInputNoCall += numVariantInputNoCall;
        numInputRef += numVariantInputRef;

        if(runMode == RunMode.Train && !getVariantIsFilterable(variantContext, applySampleAlleleCounts)) {
            // no need to train on unfilterable variants
            return;
        }
        ++numVariants;

        double alleleFrequency = variantContext.getAttributeAsDouble(VCFConstants.ALLELE_FREQUENCY_KEY, -1.0);
        if(alleleFrequency <= 0) {
            if(variantContext.getNSamples() <= minSamplesToEstimateAlleleFrequency) {
                throw new GATKException("VCF does not have " + VCFConstants.ALLELE_FREQUENCY_KEY + " annotated or enough samples to estimate it ("
                                        + minSamplesToEstimateAlleleFrequencyKey + "=" + minSamplesToEstimateAlleleFrequency + " but there are "
                                        + variantContext.getNSamples() + " samples)");
            }
            // VCF not annotated with allele frequency, guess it from allele counts
            alleleFrequency = numNonRefAlleles / (double) numCalledAlleles;
        }
        propertiesTable.append(VCFConstants.ALLELE_FREQUENCY_KEY, alleleFrequency);

        final String svType = variantContext.getAttributeAsString(VCFConstants.SVTYPE, null);
        if(svType == null) {
            throw new GATKException("Missing " + VCFConstants.SVTYPE + " for variant " + variantContext.getID());
        }
        propertiesTable.append(VCFConstants.SVTYPE, svType);

        final int svLen = variantContext.getAttributeAsInt(SVLEN_KEY, Integer.MIN_VALUE);
        if(svLen == Integer.MIN_VALUE) {
            throw new GATKException("Missing " + SVLEN_KEY + " for variant " + variantContext.getID());
        }
        propertiesTable.append(SVLEN_KEY, svLen);

        propertiesTable.append(FILTER_KEY, variantContext.getFilters());

        final Set<String> vcEvidence = Arrays.stream(
                    variantContext.getAttributeAsString(EVIDENCE_KEY, NO_EVIDENCE)
                        .replaceAll("[\\[\\] ]", "").split(",")
            ).map(ev -> ev.equals(".") ? NO_EVIDENCE : ev).collect(Collectors.toSet());
        if(vcEvidence.isEmpty()) {
            throw new GATKException("Missing " + EVIDENCE_KEY + " for variant " + variantContext.getID());
        }
        propertiesTable.append(EVIDENCE_KEY, vcEvidence);


        final Set<String> vcAlgorithms = Arrays.stream(
                variantContext.getAttributeAsString(ALGORITHMS_KEY, NO_ALGORITHM)
                        .replaceAll("[\\[\\] ]", "").split(",")
        ).map(ev -> ev.equals(".") ? NO_ALGORITHM : ev).collect(Collectors.toSet());
        if(vcAlgorithms.isEmpty()) {
            throw new GATKException("Missing " + ALGORITHMS_KEY + " for variant " + variantContext.getID());
        }
        propertiesTable.append(ALGORITHMS_KEY, vcAlgorithms);

        for(final TractOverlapDetector tractOverlapDetector : tractOverlapDetectors) {
            getTractProperties(tractOverlapDetector, variantContext);
        }

        final Iterable<Genotype> sampleGenotypes = variantContext.getGenotypesOrderedBy(sampleIds);
        propertiesTable.append(
            VCFConstants.GENOTYPE_QUALITY_KEY,
            StreamSupport.stream(sampleGenotypes.spliterator(), false).mapToInt(Genotype::getGQ).toArray()
        );

        Stream.of(PE_GQ_KEY, RD_GQ_KEY, SR_GQ_KEY).forEach(
            intGenotypeAttributeKey -> propertiesTable.append(
                    intGenotypeAttributeKey,
                    getGenotypeAttributeAsInt(sampleGenotypes, intGenotypeAttributeKey, MISSING_GQ_VAL)
            )
        );

        // get per-sample genotype qualities as a map indexed by sample ID
        final Map<String, Integer> sampleGenotypeQualities = variantContext.getGenotypes().stream().collect(
                Collectors.toMap(Genotype::getSampleName, Genotype::getGQ)
        );

        if(runMode == RunMode.Train) {
            variantIds.add(variantContext.getID());
            if(goodSampleVariants != null) {
                if(goodSampleVariants.containsKey(variantContext.getID())) {
                    // Get sample IDs of known good variants that are filterable
                    final String[] knownGoodSampleIds = goodSampleVariants.get(variantContext.getID())
                            .stream()
                            .filter(keepHomvar ?
                                    sampleId -> applySampleAlleleCounts.get(sampleId) == 1 :
                                    sampleId -> applySampleAlleleCounts.get(sampleId) > 0)
                            .toArray(String[]::new);
                    if(knownGoodSampleIds.length > 0) {
                        // get indices and GQ values of good samples for this variant
                        goodSampleVariantIndices.put(
                            numVariants - 1,
                            Arrays.stream(knownGoodSampleIds).map(sampleIndices::get).collect(Collectors.toSet())
                        );
                        // Get GQ values of known good variants that are filterable
                        goodVariantGqs.put(
                            numVariants - 1,
                            Arrays.stream(knownGoodSampleIds).mapToInt(sampleGenotypeQualities::get).toArray()
                        );
                    }
                }
                if(badSampleVariants.containsKey(variantContext.getID())) {
                    // Get sample IDs of known bad variants that are filterable
                    final String[] knownBadSampleIds = badSampleVariants.get(variantContext.getID())
                            .stream()
                            .filter(keepHomvar ?
                                    sampleId -> applySampleAlleleCounts.get(sampleId) == 1 :
                                    sampleId -> applySampleAlleleCounts.get(sampleId) > 0)
                            .toArray(String[]::new);
                    if(knownBadSampleIds.length > 0) {
                        // get indices and GQ values of good samples for this variant
                        badSampleVariantIndices.put(
                                numVariants - 1,
                                Arrays.stream(knownBadSampleIds).map(sampleIndices::get).collect(Collectors.toSet())
                        );
                        // Get GQ values of known good variants that are filterable
                        badVariantGqs.put(
                                numVariants - 1,
                                Arrays.stream(knownBadSampleIds).mapToInt(sampleGenotypeQualities::get).toArray()
                        );
                    }
                }
            }

            // get allele counts and no-call counts for training samples for this variant
            packedSampleVariantCounts.add(
                packSampleAlleleCounts(
                    sampleIds.stream().mapToInt(applySampleAlleleCounts::get).toArray(),
                    sampleIds.stream().mapToInt(applySampleNoCallCounts::get).toArray()
                )
            );
            // get genotype qualities for training samples for this variant
            sampleVariantGenotypeQualities.add(
                sampleIds.stream().mapToInt(sampleGenotypeQualities::get).toArray()
            );
        } else {
            packedSampleVariantCounts.add(
                packSampleAlleleCounts(
                    sampleIds.stream().mapToInt(applySampleAlleleCounts::get).toArray(),
                    sampleIds.stream().mapToInt(applySampleNoCallCounts::get).toArray()
                )
            );
            sampleVariantGenotypeQualities.add(sampleIds.stream().mapToInt(sampleGenotypeQualities::get).toArray());
            propertiesTable.validateAndFinalize();

            vcfWriter.add(filterVariantContext(variantContext));

            propertiesTable.clearRows();
            packedSampleVariantCounts.clear();
            sampleVariantGenotypeQualities.clear();
        }
    }

    Genotype[] filterGenotypes = null;

    private static List<Allele> decreaseAlleleCount(final Genotype genotype, final Allele refAllele) {
        final List<Allele> outputAlleles = new ArrayList<>(genotype.getPloidy());
        boolean madeDecrease = false;
        for(final Allele allele : genotype.getAlleles()) {
            if(madeDecrease || allele.isReference()) {
                outputAlleles.add(allele);
            } else {
                outputAlleles.add(refAllele);
                madeDecrease = true;
            }
        }
        return outputAlleles;
    }

    private VariantContext filterVariantContext(final VariantContext variantContext) {
        if(filterGenotypes == null) {
            filterGenotypes = new Genotype[variantContext.getNSamples()];
        } else {
            if(filterGenotypes.length != variantContext.getNSamples()) {
                throw new IllegalArgumentException("At " + variantContext.getID() + ": number of samples changed from "
                                                   + filterGenotypes.length + " to " + variantContext.getNSamples());
            }
        }
        final boolean maybeFilterable = !(keepMultiallelic && getIsMultiallelic(variantContext));

        int numFiltered = 0;
        int sampleIndex = 0;
        for(final Genotype genotype : variantContext.getGenotypes()) {
            final int alleleCount = getSampleVariantAlleleCounts(0, sampleIndex);
            if(maybeFilterable && alleleCountIsFilterable(alleleCount)) {
                ++numFilterableGenotypes;
                final int gq = adjustedGq(
                    propertiesTable.getPropertiesRow(0, sampleIndex, needsNormalizedProperties())
                );
                if (gq >= minGQ) {
                    filterGenotypes[sampleIndex] = new GenotypeBuilder(genotype).GQ(gq).make();
                } else {
                    filterGenotypes[sampleIndex] = new GenotypeBuilder(genotype)
                        .alleles(
                            alleleCount == 1 ?
                                GATKVariantContextUtils.noCallAlleles(genotype.getPloidy()) :
                                decreaseAlleleCount(genotype, variantContext.getReference())
                        )
                        .GQ(gq)
                        .make();
                    ++numFiltered;
                }
            } else{
                // still need to move GQ from phred scale to logit scale
                final int gq = p_to_scaled_logits(1.0 - phredToProb(genotype.getGQ()));
                filterGenotypes[sampleIndex] = new GenotypeBuilder(genotype).GQ(gq).make();
            }
            int noCallCounts = 0;
            int nonRefCounts = 0;
            for (final Allele allele : filterGenotypes[sampleIndex].getAlleles()) {
                if (allele.isNoCall()) {
                    ++noCallCounts;
                } else if (!allele.isReference()) {
                    ++nonRefCounts;
                }
            }
            if (nonRefCounts > 0) {
                ++numOutputVar;
            } else if (noCallCounts > 0) {
                ++numOutputNoCall;
            } else {
                ++numOutputRef;
            }
            ++sampleIndex;
        }
        final VariantContextBuilder variantContextBuilder = new VariantContextBuilder(variantContext)
            .genotypes(filterGenotypes).attribute(MIN_GQ_KEY, minGQ);
        if(numFiltered > variantContext.getNSamples() * reportMinGqFilterThreshold) {
            variantContextBuilder.filter(EXCESSIVE_MIN_GQ_FILTER_KEY);
        }

        numFilteredGenotypes += numFiltered;
        return variantContextBuilder.make();
    }


    private boolean notCloseEnough(final String strRep, final double val,
                                   @SuppressWarnings("SameParameterValue") final double tol) {
        return FastMath.abs(Double.parseDouble(strRep) - val) > FastMath.abs(tol * val);
    }

    static String padWidth(final String baseString, final int width) {
        return baseString.length() >= width ?
            baseString :
            baseString + String.join("", Collections.nCopies(width - baseString.length(), " "));
    }

    private String[] getPropertyBinNames( final PropertiesTable.Property property, final double[] bins) {
        double minVal = bins[0];
        double maxVal = bins[bins.length - 1];
        for(int variantIndex = 0; variantIndex < property.getNumRows(); ++variantIndex) {
            final float val = property.getAsFloat(variantIndex, 0, false);
            if(val < minVal) {
                minVal = val;
            } else if(val > maxVal) {
                maxVal = val;
            }
        }

        int maxWidth = property.name.length();
        final String[] propertyBinNames = new String[bins.length + 1];
        for(int i = 0; i < bins.length + 1; ++i) {
            final double low = i == 0 ? minVal : bins[i - 1];
            final double high = i == bins.length ? maxVal : bins[i];
            int precision = 0;
            String lowStr = String.format("%." + precision + "f", low);
            String highStr = String.format("%." + precision + "f", high);
            while(lowStr.equals(highStr)) {
                ++precision;
                lowStr = String.format("%." + precision + "f", low);
                highStr = String.format("%." + precision + "f", high);
            }
            int lowPrecision = precision;
            while(notCloseEnough(lowStr, low, 0.1)) {
                ++lowPrecision;
                lowStr = String.format("%." + lowPrecision + "f", low);
            }
            int highPrecision = precision;
            while(notCloseEnough(highStr, high, 0.1)) {
                ++highPrecision;
                highStr = String.format("%." + highPrecision + "f", high);
            }
            propertyBinNames[i] = lowStr + "-" + highStr;
            maxWidth = max(maxWidth, propertyBinNames[i].length());
        }
        // pad bin names so they are all the same length, for easy table viewing
        for(int i = 0; i < bins.length + 1; ++i) {
            propertyBinNames[i] = padWidth(propertyBinNames[i], maxWidth);
        }

        return propertyBinNames;
    }

    private void setPropertyBins() {
        propertyBins = new int[numVariants]; // guaranteed to be all zeros by Java spec

        // Bin variants by SVTYPE
        propertyBinLabelColumns = new ArrayList<>(1 + propertyBinsMap.size());

        propertyBinLabelColumns.add(VCFConstants.SVTYPE);
        final PropertiesTable.StringArrProperty svType =
            (PropertiesTable.StringArrProperty) propertiesTable.get(VCFConstants.SVTYPE);
        final List<String> allSvTypes = svType.getAllLabels();

        final Map<String, Integer> binNames = IntStream.range(0, allSvTypes.size())
            .boxed()
            .collect(Collectors.toMap(allSvTypes::get, i -> i));
        IntStream.range(0, numVariants).forEach(variantIndex ->
            propertyBins[variantIndex] = binNames.get(svType.getAsString(variantIndex))
        );

        // Successively refine bins by each element of propertyBinsMap
        for(Map.Entry<String, double[]> propertyBinsEntry : propertyBinsMap.entrySet()) {
            final String propertyName = propertyBinsEntry.getKey();
            final double[] bins = propertyBinsEntry.getValue();

            // get the old bin names as an ordered list, so that existing propertyBins map correctly to oldBinNames
            final List<String> oldBinNames = binNames.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .collect(Collectors.toList());
            binNames.clear();

            // get the property values, and strings representing the bins we'll use
            final PropertiesTable.Property property = propertiesTable.get(propertyName);
            final String[] propertyBinNames = getPropertyBinNames(property, bins);
            // add property name to the header with extra padding if needed
            propertyBinLabelColumns.add(padWidth(propertyName, propertyBinNames[0].length()));

            IntStream.range(0, numVariants).forEach(variantIndex -> {
                // for each variant, find which of the new bins it should index into
                int propertyBin = Arrays.binarySearch(bins, property.getAsFloat(variantIndex, 0, false));
                if(propertyBin < 0) { propertyBin = ~propertyBin; }
                // form the new bin name by adding a new column with the appropriate binned value
                final String newBinName = oldBinNames.get(propertyBins[variantIndex]) + "\t" + propertyBinNames[propertyBin];
                if(binNames.containsKey(newBinName)) {  // this bin exists, just get its index
                    propertyBins[variantIndex] = binNames.get(newBinName);
                } else {  // this is a new bin, create it and get its index
                    final int newBin = binNames.size();
                    binNames.put(newBinName, newBin);
                    propertyBins[variantIndex] = newBin;
                }
            });
        }

        numPropertyBins = binNames.size();
        propertyBinLabels = new String[numPropertyBins];
        binNames.forEach((key, value) -> propertyBinLabels[value] = key);

        variantsPerPropertyBin = new long[numPropertyBins];
        Arrays.stream(propertyBins).forEach(bin -> ++variantsPerPropertyBin[bin]);
        // Finally, want the property bin descriptions to be in sorted order for easier reading out fitting results
        final int[] sortInds = IntStream.range(0, numPropertyBins)
            .boxed()
            .sorted(Comparator.comparing(i -> propertyBinLabels[i]))
            .mapToInt(Integer::new)
            .toArray();
        propertyBinLabels = IntStream.range(0, numPropertyBins)
            .mapToObj(i -> propertyBinLabels[sortInds[i]])
            .toArray(String[]::new);
        variantsPerPropertyBin = IntStream.range(0, numPropertyBins)
            .mapToLong(i -> variantsPerPropertyBin[sortInds[i]])
            .toArray();

        final int[] unsortInds = new int[numPropertyBins];
        IntStream.range(0, numPropertyBins).forEach(i -> unsortInds[sortInds[i]] = i);
        propertyBins = Arrays.stream(propertyBins)
            .map(i -> unsortInds[i])
            .toArray();
    }

    private IntStream streamFilterableGq(final IntStream acStream, final IntStream gqStream) {
        final PrimitiveIterator.OfInt acIterator = acStream.iterator();
        return gqStream.filter(keepHomvar ? gc -> acIterator.nextInt() == 1 : gc -> acIterator.nextInt() > 0);
    }

    protected IntStream getCandidateMinGqs(final int variantIndex) {
        final IntStream alleleCountsStream = Arrays.stream(getTrioAlleleCountsMatrix(variantIndex))
                                                .flatMapToInt(Arrays::stream);
        final IntStream genotypeQualitiesStream = Arrays.stream(getTrioGenotypeQualitiesMatrix(variantIndex))
                                                .flatMapToInt(Arrays::stream);
        return getCandidateMinGqs(alleleCountsStream, genotypeQualitiesStream);
    }

    protected IntStream getCandidateMinGqs(final int[][] alleleCounts, final int[][] genotypeQualities) {
        final IntStream alleleCountsStream = Arrays.stream(alleleCounts).flatMapToInt(Arrays::stream);
        final IntStream genotypeQualitiesStream = Arrays.stream(genotypeQualities).flatMapToInt(Arrays::stream);
        return getCandidateMinGqs(alleleCountsStream, genotypeQualitiesStream);
    }

    protected IntStream getCandidateMinGqs(final IntStream alleleCountsStream, final IntStream genotypeQualitiesStream) {
        // form list of candidate min GQ values that could alter filtering for this variant
        final List<Integer> candidateGq = streamFilterableGq(alleleCountsStream, genotypeQualitiesStream)
                .distinct()
                .sorted()
                .boxed()
                .collect(Collectors.toList());
        if(candidateGq.isEmpty()) {
            return null;
        }
        // consider filtering out all filterable variants by adding 1 to highest filterable GQ value
        candidateGq.add(1 + candidateGq.get(candidateGq.size() - 1));

        // These candidate values are as large as they can be without altering filtering results.
        // If possible, move candidates down (midway to next change value) so that inaccuracy in predicting
        // candidateMinGq has tolerance in both directions.
        for(int index = candidateGq.size() - 2; index >= 0; --index) {
            final int gq = candidateGq.get(index);
            final int delta_gq = index == 0 ?
                    2:
                    gq - candidateGq.get(index - 1);
            if(delta_gq > 1) {
                candidateGq.set(index, gq - delta_gq / 2);
            }
        }

        // return stream of primitive int
        return candidateGq.stream().mapToInt(Integer::intValue);
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean maybeMendelian(final int fatherAc, final int motherAc, final int childAc,
                                   final int fatherNoCalls, final int motherNoCalls, final int childNoCalls) {
        if(fatherNoCalls > 0 || motherNoCalls > 0 || childNoCalls > 0) {
            return false; // don't try to figure out what happened if there are no-calls
        } else if(keepHomvar) {
            // since homvar can't be filtered, there are some allele count combos that can't ever be Mendelian
            if(strictMendelian) {
                final int minAc = fatherAc / 2 + motherAc / 2;
                if(childAc < minAc) {
                    return false;
                } else if(childAc == 2 && (fatherAc == 0 || motherAc == 0)) {
                    return false;
                }
            } else if(childAc == 2 && (fatherAc == 0 && motherAc == 0)) {
                return false;
            }
        }
        return true;
    }

    private boolean isMendelian(final int fatherAc, final int motherAc, final int childAc,
                                final int fatherNoCalls, final int motherNoCalls, final int childNoCalls) {
        if(fatherNoCalls > 0 || motherNoCalls > 0 || childNoCalls > 0) {
            return false; // don't try to figure out what happened if there are no-calls
        } else if(strictMendelian) {
            // child allele counts should not exhibit de-novo mutations nor be missing inherited homvar
            final int maxAc = (fatherAc > 0 ? 1 : 0) + (motherAc > 0 ? 1 : 0);
            final int minAc = fatherAc / 2 + motherAc / 2;
            return (minAc <= childAc) && (childAc <= maxAc);
        } else {
            // child allele counts should not exhibit de-novo mutations
            final int maxAc = fatherAc > 0 || motherAc > 0 ? 2 : 0;
            return childAc <= maxAc;
        }
    }

    private double getPMendelian(final int fatherAc, final int motherAc, final int childAc,
                                 final int fatherNoCalls, final int motherNoCalls, final int childNoCalls,
                                 final float pFather, final float pMother, final float pChild,
                                 final double[] d1P) {
        // Note:
        //  -don't try to untangle trios with no-calls: pMendelian = 0 and so are the derivatives
        //  -inherit 0 or 1 alleles from each parent so pInherit0 = 1 - pInherit1
        //  -similarly dPInherit0 = -dPInherit1
        //  -diagonal of Hessian is always zero, so don't need d2P
        //  -assume no-calls have the same distribution of variant/ref as calls (wild guess)
        if(fatherNoCalls > 0 || motherNoCalls > 0 || childNoCalls > 0) {
            d1P[FATHER_IND] = 0;
            d1P[MOTHER_IND] = 0;
            d1P[CHILD_IND] = 0;
            return 0.0;
        }

        final double dPInherit1Father = fatherAc / 2.0;
        final double pInherit1Father = dPInherit1Father * pFather;
        final double pInherit0Father = 1.0 - pInherit1Father;

        final double dPInherit1Mother = motherAc / 2.0;
        final double pInherit1Mother = dPInherit1Mother * pMother;
        final double pInherit0Mother = 1.0 - pInherit1Mother;

        final double pInheritRef = pInherit0Father * pInherit0Mother;
        switch(childAc) {
            case 0: {

                d1P[FATHER_IND] = -dPInherit1Father * pInherit0Mother;
                d1P[MOTHER_IND] = -dPInherit1Mother * pInherit0Father;
                d1P[CHILD_IND] = 0.0;  // pChild is irrelevant
                return pInheritRef;
            }
            case 1: {
                final double pInheritHet = pInherit1Father * pInherit0Mother + pInherit0Father * pInherit1Mother;
                d1P[FATHER_IND] = dPInherit1Father * (pChild * (pInherit0Mother - pInherit1Mother)
                                                      - (1.0 - pChild) * pInherit0Mother);
                d1P[MOTHER_IND] = dPInherit1Mother * (pChild * (pInherit0Father - pInherit1Father)
                                                      - (1.0 - pChild) * pInherit0Father);
                d1P[CHILD_IND] = (pInheritHet - pInheritRef);
                return pChild * pInheritHet + (1.0 - pChild) * pInheritRef;
            }
            case 2:
                final double pInheritHomVar = pInherit1Father * pInherit1Mother;
                if(keepHomvar) {
                    d1P[FATHER_IND] = dPInherit1Father * pInherit1Mother;
                    d1P[MOTHER_IND] = dPInherit1Mother * pInherit1Father;
                    d1P[CHILD_IND] = 0.0;  //pChild is irrelevant
                    return pInheritHomVar;
                } else {
                    d1P[FATHER_IND] = dPInherit1Father * (pChild * pInherit1Mother - (1.0 - pChild) * pInherit0Mother);
                    d1P[MOTHER_IND] = dPInherit1Mother * (pChild * pInherit1Father - (1.0 - pChild) * pInherit0Father);
                    d1P[CHILD_IND] = (pInheritHomVar - pInheritRef);
                    return pChild * pInheritHomVar + (1.0 - pChild) * pInheritRef;
                }
            default:
                throw new IllegalArgumentException("Child ploidy is greater than two.");
        }
    }

    final void setSampleIndexToPredictionIndex(final Integer[] sampleIndexToPredictionIndex,
                                               final int[] sampleAlleleCounts,
                                               final Set<Integer> goodSampleIndices,
                                               final Set<Integer> badSampleIndices,
                                               final int startPredictionIndex) {
        int predictionIndex = startPredictionIndex;
        for(int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
            if(alleleCountIsFilterable(sampleAlleleCounts[sampleIndex]) &&
                    (allTrioSampleIndices.contains(sampleIndex) ||
                            goodSampleIndices.contains(sampleIndex) ||
                            badSampleIndices.contains(sampleIndex))
            ) {
                sampleIndexToPredictionIndex[sampleIndex] = predictionIndex;
                ++predictionIndex;
            } else {
                sampleIndexToPredictionIndex[sampleIndex] = null;
            }
        }
    }

    protected float logitsToProb(final float predict) {
        return 1.0F / (1.0F + (float)FastMath.exp(-predictionScaleFactor * predict));
    }

    protected float probToLogits(final float sigma) {
        return (float)(FastMath.log(sigma / (1F - sigma)) / predictionScaleFactor);
    }


    /**
     * Convert from dLoss / dProb to dLoss / dLogits (assumes prob = sigma(logits))
     * @param prob
     * @param d1LossDProb
     * @param d2LossDProb
     * @param weight
     * @param d1LossDLogits
     * @param d2LossDLogits
     * @param sampleIndex
     */
    protected void setLossDerivs(final double prob, final double d1LossDProb, final double d2LossDProb,
                                 final double weight, final double[] d1LossDLogits, final double[] d2LossDLogits,
                                 final int sampleIndex) {
        final double d1ProbDLogit = prob * (1.0 - prob) * predictionScaleFactor;
        final double d2ProbDLogit = d1ProbDLogit * (1.0 - 2.0 * prob) * predictionScaleFactor;
        d1LossDLogits[sampleIndex] += weight * d1LossDProb * d1ProbDLogit;
        d2LossDLogits[sampleIndex] += weight * (d2LossDProb * d1ProbDLogit * d1ProbDLogit
                                                + d1LossDProb * d2ProbDLogit);
    }

    protected void setLossDerivs(final double prob, final double d1LossDProb, final double d2LossDProb,
                                 final double weight, final float[] d1LossDLogits, final float[] d2LossDLogits,
                                 final int sampleIndex) {
        final double d1ProbDLogit = prob * (1.0 - prob) * predictionScaleFactor;
        final double d2ProbDLogit = d1ProbDLogit * (1.0 - 2.0 * prob) * predictionScaleFactor;
        d1LossDLogits[sampleIndex] = (float)(d1LossDLogits[sampleIndex] + weight * d1LossDProb * d1ProbDLogit);
        d2LossDLogits[sampleIndex] = (float)(d2LossDLogits[sampleIndex]
                                             + weight * (d2LossDProb * d1ProbDLogit * d1ProbDLogit
                                                        + d1LossDProb * d2ProbDLogit));
    }

    private double getInheritancePrecision(final float[] pSample, final int[] sampleAlleleCounts,
                                           final int[] sampleNoCallCounts,
                                           final Integer[] sampleIndexToPredictionIndex,
                                           final int variantIndex,
                                           final double[] d1Precision) {
        final double[] d1PMendelianDPSample = new double[3];
        final long numMaybeMendelianTrios = maxDiscoverableMendelianTrioCount[variantIndex];
        if(numMaybeMendelianTrios == 0) {
            return Double.NaN;
        }
        double numExpectedMendelianTrios = 0.0;
        for(int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
            final int[] sampleIndices = trioSampleIndices[trioIndex];
            final int fatherSampleIndex = sampleIndices[FATHER_IND];
            final int motherSampleIndex = sampleIndices[MOTHER_IND];
            final int childSampleIndex = sampleIndices[CHILD_IND];
            final int fatherAc = sampleAlleleCounts[fatherSampleIndex];
            final int motherAc = sampleAlleleCounts[motherSampleIndex];
            final int childAc = sampleAlleleCounts[childSampleIndex];
            final int fatherNoCalls = sampleNoCallCounts[fatherSampleIndex];
            final int motherNoCalls = sampleNoCallCounts[motherSampleIndex];
            final int childNoCalls = sampleNoCallCounts[childSampleIndex];
            if(!maybeMendelian(fatherAc, motherAc, childAc, fatherNoCalls, motherNoCalls, childNoCalls)) {
                continue;  // don't assess inheritance accuracy for this trio
            }
            final Integer fatherPredictionIndex = sampleIndexToPredictionIndex[fatherSampleIndex];
            final Integer motherPredictionIndex = sampleIndexToPredictionIndex[motherSampleIndex];
            final Integer childPredictionIndex = sampleIndexToPredictionIndex[childSampleIndex];
            final float pFather = fatherPredictionIndex == null ? 1F : pSample[fatherPredictionIndex];
            final float pMother = motherPredictionIndex == null ? 1F : pSample[motherPredictionIndex];
            final float pChild = childPredictionIndex == null ? 1F : pSample[childPredictionIndex];

            final double pMendelian = getPMendelian(
                    fatherAc, motherAc, childAc, fatherNoCalls, motherNoCalls, childNoCalls,
                    pFather, pMother, pChild, d1PMendelianDPSample
            );
            numExpectedMendelianTrios += pMendelian;
            if(fatherPredictionIndex != null) {
                d1Precision[fatherPredictionIndex] = d1PMendelianDPSample[FATHER_IND] / numMaybeMendelianTrios;
            }
            if(motherPredictionIndex != null) {
                d1Precision[motherPredictionIndex] = d1PMendelianDPSample[MOTHER_IND] / numMaybeMendelianTrios;
            }
            if(childPredictionIndex != null) {
                d1Precision[childPredictionIndex] = d1PMendelianDPSample[CHILD_IND] / numMaybeMendelianTrios;
            }
        }
        return numExpectedMendelianTrios / numMaybeMendelianTrios;
    }

    private double getTruthPrecision(final float[] pSample, final Set<Integer> goodSampleIndices,
                                     final Set<Integer> badSampleIndices,
                                     final Integer[] sampleIndexToPredictionIndex,
                                     final int variantIndex,
                                     final double[] d1Precision) {

        final int numTruth = goodSampleIndices.size() + badSampleIndices.size();
        if(numTruth == 0) {
            return Double.NaN;
        }
        double precision = 0.0;
        for(final int goodSampleIndex : goodSampleIndices) {
            final Integer predictionIndex = sampleIndexToPredictionIndex[goodSampleIndex];
            if(predictionIndex != null) {
                precision += pSample[predictionIndex];
                d1Precision[predictionIndex] = 1.0 / numTruth;
            }
        }
        for(final int badSampleIndex : badSampleIndices) {
            final Integer predictionIndex = sampleIndexToPredictionIndex[badSampleIndex];
            if (predictionIndex != null) {
                precision += (1.0 - pSample[predictionIndex]);
                d1Precision[predictionIndex] = -1.0 / numTruth;
            }
        }
        return precision / numTruth;
    }

    final double getRecall(final float[] pSample, final double[] d1Recall) {
        double recall = 0.0;
        for(int index = 0; index < pSample.length; ++index) {
            recall += pSample[index];
            d1Recall[index] = 1.0 / pSample.length;
        }
        return recall / pSample.length;
    }


    static private double square(final double x) { return x * x; }

    final double precisionRecallToLoss(final float[] pSample, final double weight,
                                       final double precision, final double[] d1Precision,
                                       final double recall, final double[] d1Recall,
                                       final double[] d1Loss, final double[] d2Loss) {
        if(Double.isNaN(precision)) {
            return Double.NaN;
        }
        else if(precision < targetPrecision) {
            // ignore recall, optimize precision
            // loss = 1.0 - precision; d1Loss = -d1Precision
            for(int idx = 0; idx < pSample.length; ++idx) {
                setLossDerivs(pSample[idx], -d1Precision[idx], 0.0, weight, d1Loss, d2Loss, idx);
            }
            return 1.0 - precision;
        } else if(recall <= 0) {
            // fringe case of the general case below, probably can't happen realistically because p is never 0,
            // but this eliminates a conceivable source of algorithm failure
            final double omTargetPrecision = 1.0 - targetPrecision;
            for(int idx = 0; idx < pSample.length; ++idx) {
                final double d1F = d1Recall[idx];
                final double d2F = -2 * square(d1Recall[idx]) / precision;

                setLossDerivs(pSample[idx], -omTargetPrecision * d1F, -omTargetPrecision * d2F,
                        weight, d1Loss, d2Loss, idx);
            }
            return omTargetPrecision;

        } else {
            // optimize f, scaled and shifted to be match the precision-only loss at targetPrecision and 0 recall
            // loss = (1.0 - targetPrecision) * (1 - f)
            //    1/f = 1/precision + 1/recall
            final double f = 1.0 / (1.0 / precision + 1.0 / recall);
            final double omTargetPrecision = 1.0 - targetPrecision;
            for(int idx = 0; idx < pSample.length; ++idx) {
                final double d1F = square(f / precision) * d1Precision[idx]
                                 + square(f / recall) * d1Recall[idx];
                final double d2F = -2 * square(f * (d1Precision[idx] / precision - d1Recall[idx] / recall))
                                 / (recall + precision);

                setLossDerivs(pSample[idx], -omTargetPrecision * d1F, -omTargetPrecision * d2F,
                              weight, d1Loss, d2Loss, idx);
            }
            return omTargetPrecision * (1.0 - f);
        }
    }


    private FilterLoss getVariantLossInheritanceAndTruth(final float[] pSample, final double[] d1Loss,
                                                         final double[] d2Loss, final int startPredictionIndex,
                                                         final Integer[] sampleIndexToPredictionIndex,
                                                         final double variantInheritanceWeight,
                                                         final double variantTruthWeight, final int variantIndex) {
        final int[] sampleAlleleCounts = getSampleAlleleCounts(variantIndex);
        final int[] sampleNoCallCounts = getSampleNoCallCounts(variantIndex);
        final Set<Integer> goodSampleIndices = goodSampleVariantIndices.getOrDefault(variantIndex,
                                                                                     Collections.emptySet());
        final Set<Integer> badSampleIndices = badSampleVariantIndices.getOrDefault(variantIndex,
                                                                                   Collections.emptySet());
        setSampleIndexToPredictionIndex(sampleIndexToPredictionIndex, sampleAlleleCounts,
                                        goodSampleIndices, badSampleIndices, startPredictionIndex);

        final double[] d1InheritancePrecision = new double[pSample.length];
        final double inheritancePrecision = getInheritancePrecision(pSample, sampleAlleleCounts, sampleNoCallCounts,
                                                                    sampleIndexToPredictionIndex, variantIndex,
                                                                    d1InheritancePrecision);
        double[] d1Recall;
        double recall;
        if(inheritancePrecision <= targetPrecision) {
            d1Recall = null;
            recall = Double.NaN;
        } else {
            d1Recall = new double[pSample.length];
            recall = getRecall(pSample, d1Recall);
        }
        final double inheritanceLoss = precisionRecallToLoss(pSample, variantInheritanceWeight, inheritancePrecision,
                                                             d1InheritancePrecision, recall, d1Recall,
                                                             d1Loss, d2Loss);

        final double[] d1TruthPrecision = new double[pSample.length];
        final double truthPrecision = getTruthPrecision(pSample, goodSampleIndices, badSampleIndices,
                                                        sampleIndexToPredictionIndex, variantIndex, d1TruthPrecision);
        if(truthPrecision > targetPrecision && d1Recall == null) {
            d1Recall = new double[pSample.length];
            recall = getRecall(pSample, d1Recall);
        }
        final double truthLoss = precisionRecallToLoss(pSample, variantTruthWeight, truthPrecision,
                                                       d1TruthPrecision, recall, d1Recall,
                                                       d1Loss, d2Loss);

        return new FilterLoss(inheritanceLoss, truthLoss, variantInheritanceWeight, variantTruthWeight, null);
    }

    protected FilterLoss getLossInheritanceAndTruth(final float[] pSample, final float[] d1PSample,
                                                    final float[] d2PSample, final int[] variantIndices) {
        // want average loss per variant, weighted so that each type of propertyBin has equal weight
        final int[] propertyBinCounts = new int[numPropertyBins];
        Arrays.stream(variantIndices)
            .forEach(i -> ++propertyBinCounts[propertyBins[i]]);

        final double[] d1PSampleDouble = new double[numSamples];  // use doubles for intermediate computation to avoid rounding error
        final double[] d2PSampleDouble = new double[numSamples];
        final Integer[] sampleIndexToPredictionIndex = new Integer[numSamples];
        FilterLoss loss = new FilterLoss(0F, 0F, 0.0, 0.0, null);
        int startPredictionIndex = 0;
        for(final int variantIndex : variantIndices) {
            final int propertyBin = propertyBins[variantIndex];
            final double variantInheritanceWeight = propertyBinInheritanceWeights[propertyBin]
                                                  / propertyBinCounts[propertyBin];
            final double variantTruthWeight = propertyBinTruthWeights[propertyBin] / propertyBinCounts[propertyBin];
            loss = FilterLoss.add(
                loss,
                getVariantLossInheritanceAndTruth(
                    pSample, d1PSampleDouble, d2PSampleDouble, startPredictionIndex, sampleIndexToPredictionIndex,
                    variantInheritanceWeight, variantTruthWeight, variantIndex
                )
            );
            for(int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
                final Integer predictionIndex = sampleIndexToPredictionIndex[sampleIndex];
                if(predictionIndex != null) {
                    d1PSample[startPredictionIndex] = (float)d1PSampleDouble[sampleIndex];
                    d2PSample[startPredictionIndex] = (float)d2PSampleDouble[sampleIndex];
                    ++startPredictionIndex;
                }
                d1PSampleDouble[sampleIndex] = 0.0;
                d2PSampleDouble[sampleIndex] = 0.0;
            }
        }
        if(startPredictionIndex != pSample.length) {
            throw new GATKException("Didn't actually handle loss of all pSample: final startPredictionIndex = " +
                    startPredictionIndex + ", pSample.length = " + pSample.length);
        }
        return loss;
    }


    private double getAlleleFrequncy(final int variantIndex) {
        return ((PropertiesTable.DoubleArrProperty)propertiesTable.get(VCFConstants.ALLELE_FREQUENCY_KEY))
                .getAsDouble(variantIndex);
    }

    protected FilterSummary getFilterSummary(final int minGq, final int variantIndex, final String label) {
        final long numDiscoverableTrios = maxDiscoverableMendelianTrioCount[variantIndex];
        final int[][] variantAlleleCounts = getTrioAlleleCountsMatrix(variantIndex);
        final int[][] variantNoCallCounts = getTrioNoCallCountsMatrix(variantIndex);
        final int[][] variantGenotypeQualities = getTrioGenotypeQualitiesMatrix(variantIndex);
        final double variantAlleleFrequency = getAlleleFrequncy(variantIndex);
        final int [] goodGqs = goodVariantGqs == null ? null : goodVariantGqs.getOrDefault(variantIndex, null);
        final int [] badGqs = badVariantGqs == null ? null : badVariantGqs.getOrDefault(variantIndex, null);
        final int propertyBin = propertyBins[variantIndex];
        final double variantTruthWeight = propertyBinTruthWeights[propertyBin];
        final double variantInheritanceWeight = propertyBinInheritanceWeights[propertyBin];
        return getFilterSummary(minGq, numDiscoverableTrios, variantAlleleCounts, variantNoCallCounts,
                                variantGenotypeQualities, variantAlleleFrequency, goodGqs, badGqs,
                                variantTruthWeight, variantInheritanceWeight, label);
    }

    protected BinnedFilterSummaries getBinnedFilterSummary(final int minGq, final int variantIndex) {
        final int propertyBin = propertyBins[variantIndex];
        return new BinnedFilterSummaries(getFilterSummary(minGq, variantIndex, propertyBinLabels[propertyBin]),
                                         propertyBin, numPropertyBins);
    }

    private int filterAlleleCountByMinGq(final int alleleCount, final int gq, final int minGq) {
        if(keepHomvar) {
            return alleleCount == 1 && gq < minGq ? 0 : alleleCount;
        } else {
            return gq < minGq ? 0 : alleleCount;
        }
    }

    protected FilterSummary getFilterSummary(final int minGq, final long numDiscoverableTrios,
                                             final int[][] alleleCounts, final int[][] noCallCounts,
                                             final int[][] genotypeQualities, final double alleleFrequency,
                                             final int[] goodGqs, final int[] badGqs,
                                             final double variantTruthWeight, final double variantInheritanceWeight,
                                             final String label) {
        long numPassed = 0;
        long numVariants = 0;
        long numMendelianTrios = 0;
        long numTruePositives = 0;
        long numFalsePositives = 0;
        long numFalseNegatives = 0;
        long numTrueNegatives = 0;
        if(goodGqs != null) {
            for (final int gq : goodGqs) {
                if(gq >= minGq) {
                    ++numTruePositives;
                } else {
                    ++numFalseNegatives;
                }
            }
        }
        if(badGqs != null) {
            for(final int gq : badGqs) {
                if(gq >= minGq) {
                    ++numFalsePositives;
                } else {
                    ++numTrueNegatives;
                }
            }
        }

        long numMaybeMendelianTrios = 0;
        for (int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
            final int[] trioAc = alleleCounts[trioIndex];
            final int[] trioNoCalls = noCallCounts[trioIndex];
            final int[] trioGq = genotypeQualities[trioIndex];
            numVariants += Arrays.stream(trioAc).filter(ac -> ac > 0).count();

            final int fatherAc = filterAlleleCountByMinGq(trioAc[FATHER_IND], trioGq[FATHER_IND], minGq);
            final int motherAc = filterAlleleCountByMinGq(trioAc[MOTHER_IND], trioGq[MOTHER_IND], minGq);
            final int childAc = filterAlleleCountByMinGq(trioAc[CHILD_IND], trioGq[CHILD_IND], minGq);
            final int numPassedTrio = (fatherAc > 0 ? 1 : 0) + (motherAc > 0 ? 1 : 0) + (childAc > 0 ? 1 : 0);
            numPassed += numPassedTrio;
            if(numPassedTrio > 0) {
                if(maybeMendelian(trioAc[FATHER_IND], trioAc[MOTHER_IND], trioAc[CHILD_IND],
                                  trioNoCalls[FATHER_IND], trioNoCalls[MOTHER_IND], trioNoCalls[CHILD_IND])) {
                    ++numMaybeMendelianTrios;
                }
                if(isMendelian(fatherAc, motherAc, childAc,
                        trioNoCalls[FATHER_IND], trioNoCalls[MOTHER_IND], trioNoCalls[CHILD_IND])) {
                    ++numMendelianTrios;
                }
            }
        }
        if(numMaybeMendelianTrios != numDiscoverableMendelianTrios) {
            throw new IllegalArgumentException("numMaybeMendelianTrios=" + numMaybeMendelianTrios +
                                               ", but numDiscoverableMendelianTrios=" + numDiscoverableMendelianTrios);
        }

        return new FilterSummary(minGq, numMendelianTrios, numDiscoverableMendelianTrios, numPassed, numVariants,
                                 numTruePositives, numFalsePositives, numFalseNegatives, numTrueNegatives,
                                 alleleFrequency > maxInheritanceAf,
                                 variantTruthWeight, variantInheritanceWeight, label);
    }

    private int getFilteredAlleleCounts(final int sampleIndex, final int[] sampleAlleleCounts,
                                        final Integer[] sampleIndexToPassIndex, final boolean[] passFilter) {
        final int vcfAlleleCounts = sampleAlleleCounts[sampleIndex];
        final Integer passIndex = sampleIndexToPassIndex[sampleIndex];
        return passIndex == null || passFilter[passIndex] ? vcfAlleleCounts : 0;
    }

    protected FilterSummary getFilterSummary(final boolean[] samplePasses, final int variantIndex, final int propertyBin) {
        long numPassed = 0;
        long numVariants = 0;
        long numMendelian = 0;
        long numTruePositives = 0;
        long numFalsePositives = 0;
        long numFalseNegatives = 0;
        long numTrueNegatives = 0;

        final Set<Integer> goodSampleIndices = goodSampleVariantIndices.getOrDefault(variantIndex, Collections.emptySet());
        final Set<Integer> badSampleIndices = badSampleVariantIndices.getOrDefault(variantIndex, Collections.emptySet());
        final int[] sampleAlleleCounts = getSampleAlleleCounts(variantIndex);
        final int[] sampleNoCallCounts = getSampleNoCallCounts(variantIndex);
        final Integer[] sampleIndexToPredictionIndex = new Integer[numSamples];
        setSampleIndexToPredictionIndex(sampleIndexToPredictionIndex, sampleAlleleCounts,
                                        goodSampleIndices, badSampleIndices, 0);
        for(int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
            final Integer predictionIndex = sampleIndexToPredictionIndex[sampleIndex];
            if(predictionIndex == null) {
                continue;
            }
            if(goodSampleIndices.contains(sampleIndex)) {
                if(samplePasses[predictionIndex]) {
                    ++numTruePositives;
                } else {
                    ++numFalseNegatives;
                }
            } else if(badSampleIndices.contains(sampleIndex)) {
                if(samplePasses[predictionIndex]) {
                    ++numFalsePositives;
                } else {
                    ++numTrueNegatives;
                }
            }
        }

        long numMaybeMendelianTrios = 0;
        for(int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
            final int[] sampleIndices = trioSampleIndices[trioIndex];
            final int fatherIndex = sampleIndices[FATHER_IND];
            final int motherIndex = sampleIndices[MOTHER_IND];
            final int childIndex = sampleIndices[CHILD_IND];
            numVariants += (sampleAlleleCounts[fatherIndex] > 0 ? 1 : 0)
                        + (sampleAlleleCounts[motherIndex] > 0 ? 1 : 0)
                        + (sampleAlleleCounts[childIndex] > 0 ? 1 : 0);

            final int fatherAc = getFilteredAlleleCounts(fatherIndex, sampleAlleleCounts, sampleIndexToPredictionIndex,
                                                         samplePasses);
            final int motherAc = getFilteredAlleleCounts(motherIndex, sampleAlleleCounts, sampleIndexToPredictionIndex,
                                                         samplePasses);
            final int childAc = getFilteredAlleleCounts(childIndex, sampleAlleleCounts, sampleIndexToPredictionIndex,
                                                        samplePasses);
            // Note that we only consider an allele to have "passed" if it was in principle filterable:
            final int numPassedTrio = (fatherAc > 0 ? 1 : 0) + (motherAc > 0 ? 1 : 0) + (childAc > 0 ? 1 : 0);
            numPassed += numPassedTrio;
            if(numPassedTrio > 0) {
                if(maybeMendelian(sampleAlleleCounts[fatherIndex], sampleAlleleCounts[motherIndex], sampleAlleleCounts[childIndex],
                                  sampleNoCallCounts[fatherIndex], sampleNoCallCounts[motherIndex], sampleNoCallCounts[childIndex])) {
                    ++numMaybeMendelianTrios;
                }
                if (isMendelian(fatherAc, motherAc, childAc, sampleNoCallCounts[fatherIndex],
                        sampleNoCallCounts[motherIndex], sampleNoCallCounts[childIndex])) {
                    ++numMendelian;
                }
            }
        }

        final String label = propertyBinLabels[propertyBin];
        final double variantTruthWeight = propertyBinTruthWeights[propertyBin];
        final double variantInheritanceWeight = propertyBinInheritanceWeights[propertyBin];
        final long numDiscoverableTrios = maxDiscoverableMendelianTrioCount[variantIndex];
        if(numMaybeMendelianTrios != numDiscoverableMendelianTrios) {
            throw new IllegalArgumentException("numMaybeMendelianTrios=" + numMaybeMendelianTrios +
                    ", but numDiscoverableMendelianTrios=" + numDiscoverableMendelianTrios);
        }
        final boolean isLargeAlleleFrequency = getAlleleFrequncy(variantIndex) > maxInheritanceAf;
        return new FilterSummary(minGQ, numMendelian, numDiscoverableTrios, numPassed, numVariants,
                numTruePositives, numFalsePositives, numFalseNegatives, numTrueNegatives,
                                 isLargeAlleleFrequency, variantTruthWeight, variantInheritanceWeight, label);
    }

    protected BinnedFilterSummaries getBinnedFilterSummary(final boolean[] samplePasses, final int variantIndex) {
        final int propertyBin = propertyBins[variantIndex];
        return new BinnedFilterSummaries(getFilterSummary(samplePasses, variantIndex, propertyBin),
                                         propertyBin, numPropertyBins);
    }

    static protected class FilterQuality extends FilterSummary implements Comparable<FilterQuality> {
        final FilterLoss loss;

        FilterQuality(final FilterSummary filterSummary) {
            super(filterSummary);
            this.loss = new FilterLoss(filterSummary);
        }

        @Override
        public int compareTo(final @NotNull MinGqVariantFilterBase.FilterQuality other) {
            final int lossCompare = this.loss.compareTo(other.loss);
            // for two equivalent-loss filters, take the more permissive one
            return lossCompare == 0 ? Integer.compare(this.minGq, other.minGq) : lossCompare;
        }

        static final FilterQuality EMPTY = new FilterQuality(FilterSummary.EMPTY);
    }

    protected FilterQuality getOptimalVariantMinGq(final int variantIndex, final FilterSummary backgroundFilterSummary) {
        final IntStream candidateMinGqs = getCandidateMinGqs(variantIndex);
        if(candidateMinGqs == null) {
            // minGq doesn't matter for this row, so return previous optimal filter or trivial filter
            return backgroundFilterSummary == null ? FilterQuality.EMPTY : new FilterQuality(backgroundFilterSummary);
        }

        final long numDiscoverable = maxDiscoverableMendelianTrioCount[variantIndex];
        final int[][] variantAlleleCounts = getTrioAlleleCountsMatrix(variantIndex);
        final int[][] variantNoCallCounts = getTrioNoCallCountsMatrix(variantIndex);
        final int[][] variantGenotypeQualities = getTrioGenotypeQualitiesMatrix(variantIndex);
        final double variantAlleleFrequency = getAlleleFrequncy(variantIndex);
        final int [] goodGqs = goodVariantGqs == null ? null : goodVariantGqs.getOrDefault(variantIndex, null);
        final int [] badGqs = badVariantGqs == null ? null : badVariantGqs.getOrDefault(variantIndex, null);
        final String label = propertyBinLabels[propertyBins[variantIndex]];
        if(backgroundFilterSummary == null) {
            // doing optimization only considering loss of each individual variant
            return candidateMinGqs.parallel()
                    .mapToObj(minGq -> new FilterQuality(
                            getFilterSummary(minGq, numDiscoverable, variantAlleleCounts, variantNoCallCounts,
                                             variantGenotypeQualities, variantAlleleFrequency, goodGqs, badGqs,
                                             truthWeight, 1.0 - truthWeight, label)
                        ))
                    .min(FilterQuality::compareTo)
                    .orElseThrow(RuntimeException::new);
        } else {
            // doing optimization considering overall loss
            return candidateMinGqs
                    .parallel()
                    .mapToObj(minGq -> new FilterQuality(
                            getFilterSummary(minGq, numDiscoverable, variantAlleleCounts, variantNoCallCounts,
                                             variantGenotypeQualities, variantAlleleFrequency, goodGqs, badGqs,
                                             truthWeight, 1.0 - truthWeight, label)
                            .add(backgroundFilterSummary)
                    ))
                    .min(FilterQuality::compareTo)
                    .orElseThrow(RuntimeException::new);
        }
    }

    private void setMaxDiscoverableMendelianTrioCount() {
        maxDiscoverableMendelianTrioCount = new int [numVariants];
        numDiscoverableMendelianTrios = 0;
        for(int variantIndex = 0; variantIndex < numVariants; ++variantIndex) {
            final int[][] variantAlleleCounts = getTrioAlleleCountsMatrix(variantIndex);
            final int[][] variantNoCallCounts = getTrioNoCallCountsMatrix(variantIndex);
            for(int trioIndex = 0; trioIndex < numTrios; ++trioIndex) {
                final int[] trioAlleleCounts = variantAlleleCounts[trioIndex];
                final int[] trioNoCallCounts = variantNoCallCounts[trioIndex];
                if(maybeMendelian(trioAlleleCounts[FATHER_IND], trioAlleleCounts[MOTHER_IND], trioAlleleCounts[CHILD_IND],
                                  trioNoCallCounts[FATHER_IND], trioNoCallCounts[MOTHER_IND], trioNoCallCounts[CHILD_IND])) {
                    ++maxDiscoverableMendelianTrioCount[variantIndex];
                }
            }
            numDiscoverableMendelianTrios += maxDiscoverableMendelianTrioCount[variantIndex];
        }
    }

    private void setPerVariantOptimalMinGq() {
        // Get initial optimal filter qualities, optimizing each variant separately
        // Collect total summary stats, store min GQ
        final List<List<Integer>> indicesList = IntStream.range(0, numPropertyBins)
            .mapToObj(propertyBin -> new ArrayList<Integer>())
            .collect(Collectors.toList());
        IntStream.range(0, numVariants)
            .forEach(index -> indicesList.get(propertyBins[index]).add(index));

        perVariantOptimalMinGq = new int[numVariants];
        propertyBinInheritanceWeights = new double[numPropertyBins];
        propertyBinTruthWeights = new double[numPropertyBins];

        FilterSummary unbinnedFilterSummary = FilterSummary.EMPTY;
        for(int propertyBin = 0; propertyBin < numPropertyBins; ++propertyBin) {
            final List<Integer> indices = indicesList.get(propertyBin);
            final FilterQuality binFilterQuality = setBinOptimalMinGq(indices, propertyBin);
            unbinnedFilterSummary = unbinnedFilterSummary.add(binFilterQuality);
            // set weight for truth and inheritance for this bin based on the loss / performance
            final double inheritanceLoss = binFilterQuality.loss.inheritanceLoss;
            propertyBinInheritanceWeights[propertyBin] = inheritanceLoss < 0.5 ? 1.0 - 2 * inheritanceLoss : 0.0;
            final double truthLoss = binFilterQuality.loss.truthLoss;
            propertyBinTruthWeights[propertyBin] = truthLoss < 0.5 ? 1.0 - 2 * truthLoss : 0.0;
        }

        if(printProgress > 0) {
            System.out.format("Done setting optimal minGQ.\n");
        }

        // scale weights so that they sum to 1.0
        final double totalInherit = Arrays.stream(propertyBinInheritanceWeights).sum();
        if(printProgress > 0) {
            System.out.format("totalInherit = %f\n", totalInherit);
        }
        propertyBinInheritanceWeights = Arrays.stream(propertyBinInheritanceWeights)
            .map(w -> (1 - truthWeight) * w / totalInherit)
            .toArray();
        if(printProgress > 0) {
            System.out.format("got propertyBinInheritanceWeights\n");
        }
        final double totalTruth = Arrays.stream(propertyBinTruthWeights).sum();
        if(printProgress > 0) {
            System.out.format("totalTruth = %f\n", totalTruth);
        }
        propertyBinTruthWeights = Arrays.stream(propertyBinTruthWeights)
            .map(w -> truthWeight * w / totalTruth)
            .toArray();
        if(printProgress > 0) {
            System.out.format("got propertyBinTruthWeights\n");
        }

        // compute the proportion of filterable variants that passed the optimal minGQ filter
        final long numFilterable = IntStream.range(0, numVariants)
                .mapToLong(
                        variantIndex -> IntStream.range(0, numSamples)
                                .filter(sampleIndex -> getSampleVariantIsFilterable(variantIndex, sampleIndex))
                                .count()
                )
                .sum();
        if(printProgress > 0) {
            System.out.format("%d filterable variant x sample genotypes\n", numFilterable);
        }

        optimalProportionOfSampleVariantsPassing = unbinnedFilterSummary.numPassed / (double)numFilterable;
        if(printProgress > 0) {
            System.out.format("Optimal proportion of variants passing: %.3f\n", optimalProportionOfSampleVariantsPassing);
        }
    }

    FilterQuality setBinOptimalMinGq(final List<Integer> indices, final int propertyBin) {
        System.out.println(
            "Optimizing minGqs for\t" + String.join("\t", propertyBinLabelColumns) + "\n" +
            "                     \t" + propertyBinLabels[propertyBin]
        );
        System.out.println();
        FilterSummary overallSummary = FilterSummary.EMPTY;
        final FilterSummary[] filterSummaries = new FilterSummary[indices.size()];
        for(int i = 0; i < indices.size(); ++i) {
            final int variantIndex = indices.get(i);
            final FilterQuality greedyFilter = getOptimalVariantMinGq(variantIndex, null);
            filterSummaries[i] = greedyFilter;
            overallSummary = overallSummary.add(greedyFilter);
            perVariantOptimalMinGq[variantIndex] = greedyFilter.minGq;
        }

        // Iteratively improve filters, optimizing for OVERALL loss
        boolean anyImproved = true;
        int numIterations = 0;
        FilterLoss previousLoss = new FilterLoss(overallSummary);
        System.out.println("           \t" + FilterLoss.getHeader(propertyBinLabelColumns));
        while(anyImproved) {
            ++numIterations;
            anyImproved = false;
            for(int i = 0; i < indices.size(); ++i) {
                final int variantIndex = indices.get(i);
                final FilterSummary previousFilter = filterSummaries[i];
                final FilterSummary backgroundFilter = overallSummary.subtract(previousFilter);
                final FilterQuality greedyFilter = getOptimalVariantMinGq(variantIndex, backgroundFilter);
                perVariantOptimalMinGq[variantIndex] = greedyFilter.minGq;
                filterSummaries[i] = greedyFilter.subtract(backgroundFilter);
                overallSummary = greedyFilter;

                if(greedyFilter.loss.lt(previousLoss)) {
                    anyImproved = true;
                } else if(greedyFilter.loss.gt(previousLoss)) {
                    throw new GATKException("Loss increased. This is a bug. previousLoss=" + previousLoss
                            + ", greedyFilter.loss=" + greedyFilter.loss);
                }
                previousLoss = greedyFilter.loss;
            }
            System.out.println("Iteration " + numIterations + "\t" + previousLoss);
        }
        displayHistogram("Optimal minGq histogram", Arrays.stream(perVariantOptimalMinGq), true);
        return new FilterQuality(overallSummary);
    }

    protected long getNumTrainableSamples(final int variantIndex) {
        return IntStream.range(0, getNumSamples())
                .filter(sampleIndex -> getSampleVariantIsTrainable(variantIndex, sampleIndex))
                .count();
    }
    /**
     * Get number of rows, account for the fact that unfilterable (e.g. already HOMREF) samples will not be used
     */
    protected long getNumTrainableSampleVariants(final int[] variantIndices) {
        return Arrays.stream(variantIndices)
            .mapToLong(this::getNumTrainableSamples)
            .sum();
    }

    /**
     *
     * @param variantIndices
     * @return
     */
    protected boolean[] getSampleVariantTruth(final int[] variantIndices) {
        final int numRows = (int) getNumTrainableSampleVariants(variantIndices);
        final boolean[] sampleVariantTruth = new boolean[numRows];

        int flatIndex = 0;
        final PropertiesTable.Property gqMat = propertiesTable.get(VCFConstants.GENOTYPE_QUALITY_KEY);
        for(final int variantIndex : variantIndices) {
            final int minGq = perVariantOptimalMinGq[variantIndex];
            for(int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
                if(!getSampleVariantIsTrainable(variantIndex, sampleIndex)) {
                    continue;
                }
                final int gq = gqMat.getAsInt(variantIndex, sampleIndex);
                sampleVariantTruth[flatIndex] = gq >= minGq;
                ++flatIndex;
            }
        }
        return sampleVariantTruth;
    }

    @SuppressWarnings("SameParameterValue")
    protected void displayHistogram(final String description, final IntStream intStream, boolean binValues) {
        final Map<Integer, Integer> rawValuesMap = new HashMap<>();
        intStream.forEach(gq -> {
            if (rawValuesMap.containsKey(gq)) {
                rawValuesMap.put(gq, 1 + rawValuesMap.get(gq));
            } else {
                rawValuesMap.put(gq, 1);
            }
        });
        if(rawValuesMap.size() == 0) {
            System.out.println(description + ": no data");
        }
        final int minGqValue = rawValuesMap.keySet().stream().min(Integer::compareTo).orElseThrow(RuntimeException::new);
        final int maxGqValue = rawValuesMap.keySet().stream().max(Integer::compareTo).orElseThrow(RuntimeException::new);

        final Map<Integer, Integer> displayValuesMap;
        if(binValues) {
            displayValuesMap = new HashMap<>();
            rawValuesMap.forEach((gq, numGq) -> {
                final int binGq;
                if (gq == 0) {
                    binGq = gq;
                } else {
                    final int magnitude = (int) Math.pow(10.0, Math.floor(Math.log10(Math.abs(gq))));
                    binGq = magnitude * (gq / magnitude);
                }

                if (displayValuesMap.containsKey(binGq)) {
                    displayValuesMap.put(binGq, numGq + displayValuesMap.get(binGq));
                } else {
                    displayValuesMap.put(binGq, numGq);
                }
            });
        } else {
            displayValuesMap = rawValuesMap;
        }
        System.out.println(description + ":");
        System.out.println("min=" + minGqValue + ", max=" + maxGqValue);
        displayValuesMap.keySet()
            .stream()
            .sorted()
            .forEach(minGq -> System.out.println(minGq + ": " + displayValuesMap.get(minGq)));
    }

    @SuppressWarnings("SameParameterValue")
    protected void displayHistogram(final String description, final DoubleStream doubleStream, final int numBins,
                                    final double minValue, final double maxValue) {
        final double safety = 100;
        final double valueEps = safety * FastMath.max(FastMath.ulp(FastMath.abs(minValue)),
                                                      FastMath.ulp(FastMath.abs(maxValue)));
        final double numBinsEps = safety * FastMath.ulp((double)numBins);
        final double valueOffset = minValue - valueEps;
        final double binScale = (numBins - numBinsEps) / (maxValue - valueOffset);
        final long[] valueBins = new long [numBins];
        double[] valueRange = new double[] {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY};
        doubleStream.forEach(value -> {
            final int bin = (int)FastMath.floor((value - valueOffset) * binScale);
            try {
                ++valueBins[bin];
            } catch(ArrayIndexOutOfBoundsException arrayIndexOutOfBoundsException) {
                final String errStr = String.format("bad bin: min: %f, max: %f, scale: %f, offset: %f, value: %f\n",
                                                    minValue, maxValue, binScale, valueOffset, value);
                throw new GATKException(errStr, arrayIndexOutOfBoundsException);
            }
            if(value < valueRange[0]) {
                valueRange[0] = value;
            }
            if(value > valueRange[1]) {
                valueRange[1] = value;
            }
        });
        final long numValues = Arrays.stream(valueBins).sum();
        if(valueRange[0] > valueRange[1]) {
            System.out.println(description + ": no data");
        } else {
            System.out.format("%s: %d values\n", description, numValues);
            System.out.format("\tactual range: [%f, %f]\n", valueRange[0], valueRange[1]);
            System.out.println("\t low - high    %");
            double high = minValue;
            for(int bin = 0; bin < numBins; ++bin) {
                double low = high;
                high = low + 1.0 / binScale;
                System.out.format("\t%.2f - %.2f   %.1f\n", low, high, valueBins[bin] * 100.0 / numValues);
            }
        }
    }

    void printDebugInfo() {
        System.out.println("########################################");
        System.out.println("numVariants: " + getNumVariants());
        System.out.println("numTrios: " + getNumTrios());
        System.out.println("numProperties: " + getNumProperties());
        System.out.format("Input VCF had %.1f variants per sample.\n", numInputVar / (double)numSamples);
        System.out.println("index\tpropertyName\tpropertyBaseline\tpropertyScale");
        int idx = 0;
        for(final PropertiesTable.Property property : propertiesTable) {
            System.out.println(idx + "\t" + property.name + "\t" + property.getBaseline() + "\t" + property.getScale());
            ++idx;
        }

        final Map<String, List<String>> labelsEncoding = propertiesTable.getLabelsEncoding();
        for(final Map.Entry<String, List<String>> entry : labelsEncoding.entrySet()) {
            System.out.println(entry.getKey() + ":");
            idx = 0;
            for(final String label : entry.getValue()) {
                System.out.println(idx + "\t" + label);
                ++idx;
            }
        }

        final IntStream acStream = IntStream.range(0, getNumVariants())
            .mapToObj(this::getSampleAlleleCounts)
            .flatMapToInt(Arrays::stream);
        final IntStream gqStream = sampleVariantGenotypeQualities.stream().flatMapToInt(Arrays::stream);
        displayHistogram("Filterable alleles Gq histogram:",
                            streamFilterableGq(acStream, gqStream),true);

        System.out.println("########################################");
    }


    protected FilterLoss getLossFromPerVariantOptimization(final float[] pSampleVariantGood, final float[] d1Loss,
                                                           final float[] d2Loss, final int[] variantIndices) {
        final boolean[] sampleVariantIsGood = getSampleVariantTruth(variantIndices);
        final double[] balancedLoss = new double[numPropertyBins];
        final double[] weightedNumPredicts = new double[numPropertyBins];
        final double avgBinWeight = IntStream.range(0, numPropertyBins)
            .mapToDouble(i -> FastMath.max(propertyBinInheritanceWeights[i], propertyBinTruthWeights[i]))
            .average().orElseThrow(RuntimeException::new);
        int predictIndex = 0;
        for(final int variantIndex : variantIndices) {
            final int propertyBin = propertyBins[variantIndex];
            final int[] sampleGQs = sampleVariantGenotypeQualities.get(variantIndex);
            final int optimalMinGq = perVariantOptimalMinGq[variantIndex];
            for(int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
                if(!getSampleVariantIsTrainable(variantIndex, sampleIndex)) {
                    continue;
                }
                final double p = pSampleVariantGood[predictIndex];
                final double delta, loss, d1, d2;
                if(sampleVariantIsGood[predictIndex]) {
                    delta = FastMath.max(PROB_EPS, p);
                    loss = -FastMath.log(delta);
                    d1 = -1.0 / delta;
                    d2 = -d1 / delta;
                } else {
                    delta = FastMath.max(PROB_EPS, 1.0 - p);
                    loss = -FastMath.log(delta);
                    d1 = 1.0 / delta;
                    d2 = d1 / delta;
                }
                final double truthConfidence = max(
                    0.0,1.0  - phredToProb(FastMath.abs(sampleGQs[sampleIndex] - optimalMinGq))
                );
                final double binWeight = FastMath.max(propertyBinInheritanceWeights[propertyBin],
                                                      propertyBinTruthWeights[propertyBin])
                        / avgBinWeight;
                final double weight = truthConfidence * binWeight;
                setLossDerivs(p, d1, d2, weight, d1Loss, d2Loss, predictIndex);
                weightedNumPredicts[propertyBin] += weight;
                balancedLoss[propertyBin] += loss * weight;
                ++predictIndex;
            }
        }

        double totalWeight = 0.0;
        double totalLoss = 0.0;
        for(int propertyBin = 0; propertyBin < numPropertyBins; ++propertyBin) {
            if(weightedNumPredicts[propertyBin] == 0) {
                continue;
            }
            final double binWeight = FastMath.max(propertyBinInheritanceWeights[propertyBin],
                                                  propertyBinTruthWeights[propertyBin]);
            final double binLoss = balancedLoss[propertyBin] / weightedNumPredicts[propertyBin];
            totalWeight += binWeight;
            totalLoss += binLoss;
        }
        totalLoss /= totalWeight;
        return new FilterLoss(totalLoss, totalLoss, 1.0, 1.0, null);
    }

    private boolean[][] getSampleVariantPasses(final float[] pSampleVariantIsGood, final int[] variantIndices) {
        final boolean[][] sampleVariantPasses = new boolean[variantIndices.length][];
        int flatIndex = 0;
        for(int row = 0; row < variantIndices.length; ++row) {
            final int variantIndex = variantIndices[row];
            final boolean[] samplePasses = new boolean[(int)getNumTrainableSamples(variantIndex)];
            sampleVariantPasses[row] = samplePasses;
            for(int col = 0; col < samplePasses.length; ++col) {
                samplePasses[col] = pSampleVariantIsGood[flatIndex] >= 0.5F;
                ++flatIndex;
            }
        }
        return sampleVariantPasses;
    }

    protected void getLossMetrics(final float[] pSampleVariantIsGood, final int[] variantIndices) {
        final boolean[][] sampleVariantPasses = getSampleVariantPasses(pSampleVariantIsGood, variantIndices);

        final BinnedFilterSummaries binnedFilterSummaries =
            IntStream.range(0, variantIndices.length)
                .mapToObj(i -> getBinnedFilterSummary(sampleVariantPasses[i], variantIndices[i]))
                .reduce(BinnedFilterSummaries::add).orElse(BinnedFilterSummaries.EMPTY);
        for(final FilterSummary filterSummary : binnedFilterSummaries){
            filterSummary.toString();
        }
    }

    protected FilterLoss getLoss(final float[] pSampleVariantIsGood, final int[] variantIndices) {
        final boolean[][] sampleVariantPasses = getSampleVariantPasses(pSampleVariantIsGood, variantIndices);

        return new FilterLoss(
            IntStream.range(0, variantIndices.length)
                .mapToObj(i -> getBinnedFilterSummary(sampleVariantPasses[i], variantIndices[i]))
                .reduce(BinnedFilterSummaries::add).orElse(BinnedFilterSummaries.EMPTY),
            propertyBinLabelColumns
        );
    }

    protected FilterLoss getLoss(final int[] minGq, final int[] variantIndices) {
        if(minGq.length != variantIndices.length) {
            throw new GATKException(
                    "Length of minGq (" + minGq.length + ") does not match length of variantIndices (" + variantIndices.length + ")"
            );
        }
        return new FilterLoss(
            IntStream.range(0, minGq.length)
                .mapToObj(i -> getBinnedFilterSummary(minGq[i], variantIndices[i]))
                .reduce(BinnedFilterSummaries::add).orElse(BinnedFilterSummaries.EMPTY),
            propertyBinLabelColumns
        );
    }

    private void setTrainingAndValidationIndices() {
        final int numValidationIndices = (int)round(validationProportion * numVariants);
        final List<Integer> shuffleIndices = IntStream.range(0, numVariants).boxed().collect(Collectors.toList());
        Collections.shuffle(shuffleIndices, randomGenerator);

        validationIndices = shuffleIndices.subList(0, numValidationIndices).stream()
            .sorted().mapToInt(Integer::intValue).toArray();
        trainingIndices = shuffleIndices.subList(numValidationIndices, numVariants).stream()
            .sorted().mapToInt(Integer::intValue).toArray();
    }

    private void saveTrainedModel() {
        try (final OutputStream outputStream = modelFile.getOutputStream()) {
            final OutputStream unclosableOutputStream = new FilterOutputStream(outputStream) {
                @Override
                public void close() {
                    // don't close the stream in one of the subroutines
                }
            };
            propertiesTable.saveDataEncoding(unclosableOutputStream);
            saveModel(unclosableOutputStream);
        } catch(IOException ioException) {
            throw new GATKException("Error saving modelFile " + modelFile, ioException);
        }
    }

    private void savePropertiesTable() {
        if(propertiesTableFile == null) {
            return;
        }
        try (final OutputStream outputStream = propertiesTableFile.getOutputStream()) {
            final OutputStream unclosableOutputStream = new FilterOutputStream(outputStream) {
                @Override
                public void close() {
                    // don't close the stream in one of the subroutines
                }
            };
            propertiesTable.save(unclosableOutputStream);
        } catch(IOException ioException) {
            throw new GATKException("Error saving propertiesTable " + propertiesTableFile, ioException);
        }
    }

    private void loadTrainedModel() {
        if(modelFile == null || !Files.exists(modelFile.toPath())) {
            if(runMode == RunMode.Filter) {
                throw new UserException("mode=FILTER, but trained model file was not provided.");
            }
            return;
        }
        try (final InputStream inputStream = modelFile.getInputStream()) {
            final InputStream unclosableInputStream = new FilterInputStream(inputStream) {
                @Override
                public void close() {
                    // don't close the stream in one of the subroutines
                }
            };
            propertiesTable.loadDataEncoding(unclosableInputStream);
            loadModel(unclosableInputStream );
        } catch (Exception exception) {
            throw new GATKException("Error loading modelFile " + modelFile + " (malformed file?)", exception);
        }
        System.out.println("loadTrainedModel complete");
    }

    private byte[] modelCheckpoint = null;

    protected void saveModelCheckpoint() {
        final ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        saveModel(outputStream);
        modelCheckpoint = outputStream.toByteArray();
    }

    protected void loadModelCheckpoint() {
        final ByteArrayInputStream inputStream = new ByteArrayInputStream(modelCheckpoint);
        loadModel(inputStream);
    }

    protected abstract boolean needsNormalizedProperties();
    protected abstract float predict(final float[] sampleVariantProperties);
    protected abstract void trainFilter();
    protected abstract void saveModel(final OutputStream outputStream);
    protected abstract void loadModel(final InputStream inputStream);

    protected int phredScale(final double p) {
        return (int)FastMath.round(-10.0 * FastMath.log10(p));
    }
    protected double phredToProb(final double phred) {
        return FastMath.exp(-phredCoef * phred);
    }
    protected int adjustedGq(final float[] sampleVariantProperties) {
        return p_to_scaled_logits(predict(sampleVariantProperties));
    }

    // near p=0.5, each scaled logit is ~ 1% change in likelihood
    static final double LOGIT_SCALE = 1.0 / FastMath.log(0.51 / 0.49);

    protected int p_to_scaled_logits(final double p) {
        return p == 0 ? Integer.MIN_VALUE :
                        p == 1 ? Integer.MAX_VALUE :
                                 (int)FastMath.floor(LOGIT_SCALE * FastMath.log(p / (1.0 - p)));
    }

    protected double scaled_logits_to_p(final int scaled_logits) {
        return 1.0 / (1.0 + FastMath.exp(-scaled_logits / LOGIT_SCALE));
    }

    @Override
    public Object onTraversalSuccess() {
        if(runMode == RunMode.Train) {
            if(numVariants == 0) {
                throw new GATKException("No variants contained in vcf: " + drivingVariantFile);
            }

            if(goodSampleVariants != null) {
                // no longer needed, clear them out
                goodSampleVariants.clear();
                goodSampleVariants = null;
                badSampleVariants.clear();
                badSampleVariants = null;
            }

            setPropertyBins();
            propertiesTable.validateAndFinalize();

            savePropertiesTable();
            setTrainingAndValidationIndices();
            setMaxDiscoverableMendelianTrioCount();
            setPerVariantOptimalMinGq();

            printDebugInfo();

            trainFilter();
            saveTrainedModel();
        } else {
            System.out.println("Filter summary:");
            System.out.println("\tFiltered " + numFilteredGenotypes + " of " + numFilterableGenotypes +
                               " filterable genotypes in " + numVariants + " variants x " + numSamples + " samples");
            System.out.format("\t\t = %.1f%% of genotypes filtered.\n",
                              100 * numFilteredGenotypes / (double)(numFilterableGenotypes));
            System.out.format("\tInput VCF had %.1f variants per sample\n",
                              numInputVar / (double)numSamples);
            System.out.format("\t\t(%d ref, %d non-ref, %d no-call)\n", numInputRef, numInputVar, numInputNoCall);
            System.out.format("\tOutput VCF had %.1f variants per sample\n",
                             numOutputVar / (double)numSamples);
            System.out.format("\t\t(%d ref, %d non-ref, %d no-call)\n", numOutputRef, numOutputVar, numOutputNoCall);
        }
        return null;
    }

    @Override
    public void closeTool() {
        if (vcfWriter != null) {
            vcfWriter.close();
            System.out.println("Closed " + outputFile);
        }
    }
}
