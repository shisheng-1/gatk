import os
import logging
import numpy as np
from typing import List
from typing.io import TextIO
from DatabaseClient import BigQueryDatabaseClient, DatabaseClient

from defines import MRI_ZOOM_INPUT, MRI_ZOOM_MASK, TENSOR_MAPS_FILE_NAME, MRI_SEGMENTED_CHANNEL_MAP, DICTIONARY_TABLE, CODING_TABLE, PHENOTYPE_TABLE
from tensor_writer_ukbb import disease_prevalence_status, get_disease2tsv, disease_incidence_status, disease_censor_status

def write_tensor_maps(args) -> None:
    logging.info("Making tensor maps...")

    tensor_maps_file = f"{args.output_folder}/{TENSOR_MAPS_FILE_NAME}.py"
    db_client = BigQueryDatabaseClient(credentials_file=args.bigquery_credentials_file)
    with open(tensor_maps_file, 'w') as f:
        f.write(_get_tensor_map_file_imports())
        _write_dynamic_mri_tensor_maps(args.x, args.y, args.z, args.zoom_width, args.zoom_height, args.label_weights, args.t, f)
        _write_continuous_tensor_maps(f, db_client)
        _write_disease_tensor_maps(args.phenos_folder, f)
        _write_disease_tensor_maps_time(args.phenos_folder, f)
        _write_disease_tensor_maps_incident_prevalent(args.phenos_folder, f)
        f.write('\n')
        logging.info(f"Wrote the tensor maps to {tensor_maps_file}.")


def _get_tensor_map_file_imports() -> str:
    return f"#  TensorMaps automatically generated by {os.path.basename(__file__)}\n" \
           f"#  DO NOT EDIT\n\n" \
           f"from TensorMap import TensorMap\n" \
           f"from tensor_maps_by_hand import TMAPS \n" \
           f"from metrics import weighted_crossentropy\n" \
           f"from defines import MRI_SEGMENTED_CHANNEL_MAP\n\n\n"


def _write_dynamic_mri_tensor_maps(x: int, y: int, z: int, zoom_width: int, zoom_height: int, label_weights: List[int],
                                   t: int, f: TextIO) -> None:
    """Write TensorMappings that are adjustable from the command line to the filepath (f) provided.

    Arguments:
        :param x: MRI x dimension
        :param y: MRI y dimension
        :param z: MRI z dimension
        :param zoom_width: MRI zoomed in x dimension size
        :param zoom_height: MRI zoomed in y dimension size
        :param label_weights: Class weights for weighted cross entropy to counter class imbalance problems
        :param t: MRI time slices
        :param f: tensor_maps_by_script.py file path
    """
    if label_weights is None:
        label_weights = [20.0, 250.0, 250.0]
    f.write(f"TMAPS['mri-xyt-lax'] = TensorMap('mri-xyt-lax', ({x}, {y}, {z}), "
            f"dependent_map=TMAPS['lax-view-detect'])\n")
    f.write(f"TMAPS['mri-xyt-lax'] = TensorMap('mri-xyt-sax', ({x}, {y}, {z}), "
            f"dependent_map=TMAPS['sax-view-detect'])\n")
    f.write(f"TMAPS['mri-xyt-lax'] = TensorMap('mri-xyt-slax', ({x}, {y}, {z}), "
            f"dependent_map=TMAPS['slax-view-detect'])\n")
    f.write(f"TMAPS['{_segmented_map('mri_slice')}'] = TensorMap('{_segmented_map('mri_slice')}', "
            f"({x}, {y}, {len(MRI_SEGMENTED_CHANNEL_MAP)}), loss='categorical_crossentropy', "
            f"group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP)\n")
    f.write(f"TMAPS['{_segmented_map('mri_slice')}_weighted'] = TensorMap('{_segmented_map('mri_slice')}', ({x}, "
            f"{y}, {len(MRI_SEGMENTED_CHANNEL_MAP)}), group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP,"
            f"loss=weighted_crossentropy({label_weights}, '{_segmented_map('mri_slice')}'))\n")
    f.write(f"TMAPS['mri_slice'] = TensorMap('mri_slice', ({x}, {y}, 1), "
            f"dependent_map=TMAPS['{_segmented_map('mri_slice')}'])\n")
    f.write(f"TMAPS['mri_slice_weighted'] = TensorMap('mri_slice', ({x}, {y}, 1), "
            f"dependent_map=TMAPS['{_segmented_map('mri_slice')}_weighted'])\n")
    f.write(f"TMAPS['{_segmented_map('cine_segmented_sax_inlinevf')}'] = "
            f"TensorMap('{_segmented_map('cine_segmented_sax_inlinevf')}', ({x}, {y}, {t}, "
            f"{len(MRI_SEGMENTED_CHANNEL_MAP)}), loss='categorical_crossentropy', group='categorical', "
            f"channel_map=MRI_SEGMENTED_CHANNEL_MAP)\n")
    f.write(f"TMAPS['{_segmented_map('cine_segmented_sax_inlinevf')}_weighted'] = "
            f"TensorMap('{_segmented_map('cine_segmented_sax_inlinevf')}', ({x}, {y}, {t}, "
            f"{len(MRI_SEGMENTED_CHANNEL_MAP)}), group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP, "
            f"loss=weighted_crossentropy({label_weights}, '{_segmented_map('cine_segmented_sax_inlinevf')}'))\n")
    f.write(f"TMAPS['cine_segmented_sax_inlinevf'] = TensorMap('cine_segmented_sax_inlinevf', ({x}, {y}, "
            f"{t}, 1), dependent_map=TMAPS['{_segmented_map('cine_segmented_sax_inlinevf')}'])\n")
    f.write(f"TMAPS['cine_segmented_sax_inlinevf_weighted'] = TensorMap('cine_segmented_sax_inlinevf', "
            f"({x}, {y}, {t}, 1), "
            f"dependent_map=TMAPS['{_segmented_map('cine_segmented_sax_inlinevf')}_weighted'])\n")
    f.write(f"TMAPS['{MRI_ZOOM_MASK}'] = TensorMap('{MRI_ZOOM_MASK}', ({zoom_width},{zoom_height},{t}, "
            f"{len(MRI_SEGMENTED_CHANNEL_MAP)}), loss='categorical_crossentropy', group='categorical', "
            f"channel_map=MRI_SEGMENTED_CHANNEL_MAP)\n")
    f.write(f"TMAPS['{MRI_ZOOM_MASK}_weighted'] = TensorMap('{MRI_ZOOM_MASK}', ({zoom_width}, {zoom_height}, "
            f"{t}, {len(MRI_SEGMENTED_CHANNEL_MAP)}), group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP,"
            f"loss=weighted_crossentropy({label_weights}, '{MRI_ZOOM_MASK}'))\n")
    f.write(f"TMAPS['{MRI_ZOOM_INPUT}'] = TensorMap('{MRI_ZOOM_INPUT}',({zoom_width},{zoom_height},{t}, "
            f"1), dependent_map=TMAPS['{MRI_ZOOM_MASK}'])\n")
    f.write(f"TMAPS['{MRI_ZOOM_INPUT}_weighted'] = TensorMap('{MRI_ZOOM_INPUT}',({zoom_width},{zoom_height},{t}, "
            f"1), dependent_map=TMAPS['{MRI_ZOOM_MASK}_weighted'])\n")
    f.write(f"TMAPS['{_segmented_map('mri_systole_diastole')}']=TensorMap('{_segmented_map('mri_systole_diastole')}', "
            f"({x}, {y}, 2, {len(MRI_SEGMENTED_CHANNEL_MAP)}), loss='categorical_crossentropy', "
            f"group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP)\n")
    f.write(f"TMAPS['{_segmented_map('mri_systole_diastole')}_weighted'] = TensorMap("
            f"'{_segmented_map('mri_systole_diastole')}', ({x}, {y}, 2, {len(MRI_SEGMENTED_CHANNEL_MAP)}), "
            f"group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP,"
            f"loss=weighted_crossentropy({label_weights}, '{_segmented_map('mri_systole_diastole')}'))\n")
    f.write(f"TMAPS['mri_systole_diastole'] = TensorMap('mri_systole_diastole', ({x}, {y}, 2, 1), "
            f"dependent_map=TMAPS['{_segmented_map('mri_systole_diastole')}'])\n")
    f.write(f"TMAPS['mri_systole_diastole_weighted'] = TensorMap('mri_systole_diastole', ({x}, {y}, 2, 1), "
            f"dependent_map=TMAPS['{_segmented_map('mri_systole_diastole')}_weighted'])\n")
    f.write(f"TMAPS['{_segmented_map('mri_systole_diastole_8')}'] = TensorMap('"
            f"{_segmented_map('mri_systole_diastole_8')}', ({x}, {y}, 8, {len(MRI_SEGMENTED_CHANNEL_MAP)}), "
            f"loss='categorical_crossentropy',  group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP)\n")
    f.write(f"TMAPS['{_segmented_map('mri_systole_diastole_8')}_weighted'] = TensorMap("
            f"'{_segmented_map('mri_systole_diastole_8')}', ({x}, {y}, 8, {len(MRI_SEGMENTED_CHANNEL_MAP)}), "
            f"group='categorical', channel_map=MRI_SEGMENTED_CHANNEL_MAP,"
            f"loss=weighted_crossentropy({label_weights}, '{_segmented_map('mri_systole_diastole_8')}'))\n")
    f.write(f"TMAPS['mri_systole_diastole_8'] = TensorMap('mri_systole_diastole_8', ({x}, {y}, 8, 1), "
            f"dependent_map=TMAPS['{_segmented_map('mri_systole_diastole_8')}'])\n\n")


def _write_disease_tensor_maps(phenos_folder: str, f: TextIO)-> None:
    f.write(f"\n\n#  TensorMaps for MPG disease phenotypes\n")
    disease2tsv = get_disease2tsv(phenos_folder)
    status = disease_censor_status(disease2tsv)
    for d in sorted(list(disease2tsv.keys())):
        total = len(status[d])
        diseased = np.sum(list(status[d].values()))
        factor = int(total / (diseased * 2))
        f.write(f"TMAPS['{d}'] = TensorMap('{d}', group = 'categorical_index', channel_map = {{'no_{d}':0, '{d}':1}}, "
                f"loss = weighted_crossentropy([1.0, {factor}], '{d}'))\n")


def _write_disease_tensor_maps_incident_prevalent(phenos_folder: str, f: TextIO) -> None:
    f.write(f"\n\n#  TensorMaps for prevalent and incident MPG disease phenotypes\n")
    disease2tsv = get_disease2tsv(phenos_folder)
    status_p = disease_prevalence_status(disease2tsv, 1000000, 2000000)
    status_i = disease_incidence_status(disease2tsv, 1000000, 2000000)
    for disease in sorted(list(disease2tsv.keys())):
        total = len(status_p[disease])
        diseased_p = np.sum(list(status_p[disease].values()))
        factor_p = int(total / (1 + (diseased_p * 3)))
        diseased_i = np.sum(list(status_i[disease].values()))
        factor_i = int(total / (1 + (diseased_i * 3)))
        f.write(f"TMAPS['{disease}_prevalent_incident'] = TensorMap('{disease}', group='categorical_date', "
                f"channel_map={{'no_{disease}':0, 'prevalent_{disease}':1, 'incident_{disease}':2}}, "
                f"loss=weighted_crossentropy([1.0, {factor_p}, {factor_i}], '{disease}_prevalent_incident'))\n")


def _write_disease_tensor_maps_time(phenos_folder: str, f: TextIO) -> None:
    f.write(f"\n\n#  TensorMaps for date regression on MPG disease phenotypes\n")
    disease2tsv = get_disease2tsv(phenos_folder)
    for d in sorted(list(disease2tsv.keys())):
        f.write(f"TMAPS['{d}_time']=TensorMap('{d}',group='diagnosis_time',channel_map={{'{d}_time':0}},loss='mse')\n")

            
def _write_continuous_tensor_maps(f: TextIO, db_client: DatabaseClient):
    annotation_units = 2
    group = 'continuous'

    # Handle special coding values in continuous variables in order to generate summary statistics (mean and std dev) for
    # each field across all samples. This will remove missing samples from the calculation and change the value of 'Less than one'
    query = f"""
    WITH coding_tmp AS (
        SELECT 
            *,
            CASE
                WHEN meaning IN ('Do not know',  'Prefer not to answer', 'Ongoing when data entered') OR meaning LIKE "Still taking%" THEN TRUE
            END AS missing,
            CASE
                WHEN meaning = 'Less than one' THEN '.5'
            END AS value
        FROM
            {CODING_TABLE}
    ), pheno_tmp AS (
    SELECT 
        sample_id, 
        FieldID, 
        COALESCE(c.value, p.value) new_value, 
        COALESCE(c.missing, FALSE) missing 
    FROM {PHENOTYPE_TABLE} AS p
    LEFT JOIN coding_tmp AS c 
        ON TRUE
        AND p.value = c.coding 
        AND p.coding_file_id = c.coding_file_id
    WHERE TRUE
        AND instance = 0 
        AND array_idx = 0
    )

    SELECT 
        t.FieldID, 
        Field, 
        AVG(CAST(new_value AS FLOAT64)) mean, 
        STDDEV(CAST(new_value AS FLOAT64)) std 
    FROM pheno_tmp AS t
    LEFT JOIN {DICTIONARY_TABLE} AS d ON d.FieldID = t.FieldID
    WHERE TRUE
        AND ValueType IN ('Integer', 'Continuous') 
        AND NOT missing
    GROUP BY t.FieldID, Field 
    """

    field_data_for_tensor_maps = db_client.execute(query)

    f.write(f"\n\n#  Continuous tensor maps\n")
    for row in field_data_for_tensor_maps:
        name = str(row.FieldID) + "_" + row.Field.replace("-", "").replace(" ", "-").replace("(", "").replace(")", "")
        name = name.replace("'", "").replace(",", "").replace("/", "").replace("+", "") + "_0_0"

        f.write(f"TMAPS['{row.FieldID}_0'] = TensorMap('{name}', group='{group}', channel_map={{'{name}': 0, "
                f"'not-missing': 1}}, normalization={{'mean': {row.mean}, 'std': {row.std}}}, "
                f"annotation_units={annotation_units})\n")



def _segmented_map(name):
    return name + '_segmented'


def _get_pkl_path_for_field(field_id: int, pyukbb_data_path: str):
    """Returns the path to the .pkl file that contained `UKBioBankParsedField` for a given FieldID."""
    for _, _, files in os.walk(pyukbb_data_path):
        for file in files:
            if file.find(f'FieldID_{field_id}.pkl') > -1:
                return os.path.join(pyukbb_data_path, file)
    raise FileNotFoundError('Cannot find pyukbb .pkl file for field ID {field_id}!')


def _get_all_available_fields(available_fields_pd, keyword: str = None, category: int = None):
    filtered = available_fields_pd
    if category is not None:
        filtered = filtered[filtered.Category == category]
    if keyword is not None:
        filtered = filtered[filtered.Field.str.contains(keyword, case=False)]
    return filtered
