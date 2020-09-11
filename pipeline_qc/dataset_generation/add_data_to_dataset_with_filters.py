from lkaccess import LabKey, contexts, QueryFilter
import labkey
import logging

########################################################################################################################
"""
This script will associate all of the LabKey "Processing.Cell", "Microscopy.FOV", and "Processing.Content" database rows
found via the filter array below (which pertains to the "Processing.Cell" table) and create associations between them
and the given dataset ID.

WARNING: All existing associations for the given dataset ID will be removed before new ones are made.

Update the following variables to have the appropriate values:

"LK_ENVIRONMENT" - LabKey environment to create associations in. One of:
                     * contexts.DEV
                     * contexts.STAGE
                     * contexts.PROD  <-- The "real" LabKey
                    
"DATASET_ID" - LabKey ID number of the dataset to insert associations for

"CELLS_FILTER_ARRAY" - Filters for the "Processing.Cell" table
"""
LK_ENVIRONMENT = contexts.DEV

DATASET_ID = 121

CELLS_FILTER_ARRAY = [
    labkey.query.QueryFilter('FOVId/WellId/PlateId/PlateTypeId/Name', 'Production - Imaging', 'eq'),
    labkey.query.QueryFilter('FovEdge', '0', 'eq'),
    labkey.query.QueryFilter('FOVId/QC638nmCropBottomFalseClip', '', 'isblank'),
    labkey.query.QueryFilter('FOVId/SourceImageFileId/CellPopulationId/Clone', '67', 'contains'),
    labkey.query.QueryFilter('FOVId/QCPassIntensity', '1', 'eq'),
    labkey.query.QueryFilter('FOVId/QCBrightfieldZStacksOutOfOrder', '0', 'eq'),
    labkey.query.QueryFilter('FOVId/SourceImageFileId/CellLineId/Name', '58', 'contains')
]

########################################################################################################################
########################################################################################################################
########################################################################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def _get_existing_dataset_info(lk):
    existing_cells = lk.select_rows_as_list(
        schema_name='DataHandoff',
        query_name='DatasetCellJunction',
        filter_array=[
            QueryFilter('DatasetId', DATASET_ID)
        ]
    )
    existing_fovs = lk.select_rows_as_list(
        schema_name='DataHandoff',
        query_name='DatasetFOVJunction',
        filter_array=[
            QueryFilter('DatasetId', DATASET_ID)
        ]
    )
    existing_content = lk.select_rows_as_list(
        schema_name='DataHandoff',
        query_name='DatasetContentJunction',
        filter_array=[
            QueryFilter('DatasetId', DATASET_ID)
        ]
    )

    return existing_cells, existing_fovs, existing_content


def _delete_existing_associations(lk, existing_cells, existing_fovs, existing_content):
    if len(existing_cells):
        log.info(f"Removing {len(existing_cells)} existing Cell associations")
        lk.delete_rows_by_chunks(
            schema_name='DataHandoff',
            query_name='DatasetCellJunction',
            rows=existing_cells
        )
    else:
        log.info("No existing Cell associations found")
    if len(existing_fovs):
        log.info(f"Removing {len(existing_fovs)} existing FOV associations")
        lk.delete_rows_by_chunks(
            schema_name='DataHandoff',
            query_name='DatasetFOVJunction',
            rows=existing_fovs
        )
    else:
        log.info("No existing FOV associations found")
    if len(existing_content):
        log.info(f"Removing {len(existing_content)} existing file associations")
        lk.delete_rows_by_chunks(
            schema_name='DataHandoff',
            query_name='DatasetContentJunction',
            rows=existing_content
        )
    else:
        log.info(f"No existing file associations found")


def _get_insertable_rows(lk):
    log.info("Querying for new dataset content")
    cells = lk.select_rows_as_list(
        schema_name='processing',
        query_name='cell',
        filter_array=CELLS_FILTER_ARRAY
    )
    cell_ids = set([cell['CellId'] for cell in cells])
    fov_ids = set([cell['FOVId'] for cell in cells])

    fov_ids_str = ';'.join([str(fov) for fov in list(fov_ids)])
    content = lk.select_rows_as_list(
        schema_name='processing',
        query_name='FileFOVContent',
        filter_array=[
            QueryFilter('FOVId', fov_ids_str, QueryFilter.Types.EQUALS_ONE_OF),
            QueryFilter('ContentId/ContentGenerationAlgorithmId/Name', 'Matlab nucleus/membrane segmentation',
                        QueryFilter.Types.NOT_EQUAL)
        ]
    )
    content_ids = set([content_row['ContentId'] for content_row in content])

    dataset_cell_rows = [{'DatasetId': DATASET_ID, 'CellId': cell_id} for cell_id in cell_ids]
    dataset_fov_rows = [{'DatasetId': DATASET_ID, 'FOVId': fov_id} for fov_id in fov_ids]
    dataset_content_rows = [{'DatasetId': DATASET_ID, 'ContentId': content_id} for content_id in content_ids]

    return dataset_cell_rows, dataset_fov_rows, dataset_content_rows


def main():
    log.info(f"Populating Dataset {DATASET_ID}")
    lk = LabKey(server_context=LK_ENVIRONMENT)

    existing_cells, existing_fovs, existing_content = _get_existing_dataset_info(lk)

    _delete_existing_associations(lk, existing_cells, existing_fovs, existing_content)

    dataset_cell_rows, dataset_fov_rows, dataset_content_rows = _get_insertable_rows(lk)

    log.info(f"Creating association between Dataset {DATASET_ID} and {len(dataset_cell_rows)} cells")
    lk.insert_rows_by_chunks(
        schema_name='Datahandoff',
        query_name='DatasetCellJunction',
        rows=dataset_cell_rows
    )
    log.info(f"Creating association between Dataset {DATASET_ID} and {len(dataset_fov_rows)} FOVs")
    lk.insert_rows_by_chunks(
        schema_name='DataHandoff',
        query_name='DatasetFOVJunction',
        rows=dataset_fov_rows
    )
    log.info(f"Creating association between Dataset {DATASET_ID} and {len(dataset_content_rows)} content rows")
    lk.insert_rows_by_chunks(
        schema_name='DataHandoff',
        query_name='DatasetContentJunction',
        rows=dataset_content_rows
    )
    log.info(f"Dataset creation complete.")


if __name__ == '__main__':
    main()
