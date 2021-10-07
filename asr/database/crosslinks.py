from asr.core import ASRResult, prepare_result
from ase.db import connect
import typing


@prepare_result
class Result(ASRResult):
    """Container for crosslinks results."""

    target_db: str
    link_dbs: typing.List[str]

    key_descriptions: typing.Dict[str, str] = dict(
        target_db="DB that links get created for.",
        link_dbs="List of DB that link_db should create links for.",
    )


def main(target: str, dbs: typing.Union[str, None] = None) -> Result:
    """Create links between entries in given ASE databases."""
    # connect to target and linking DBs
    link_db = connect(target)
    db_connections = {}
    for dbfilenames in dbs:
        db = connect(dbfilenames)
        db_connections[dbfilenames] = db

    # make sure all DBs possess the correct metadata structure
    check_metadata(link_db, target, target=True)
    for dbfilename, dbconnection in db_connections.items():
        check_metadata(dbconnection, dbfilename, target=False)
    print("INFO: metadata of input DBs has the correct metadata structure.")
    print(
        "===================================================="
        "===================================================="
    )

    # for each linking DB, map uid to respective row
    uids_for_each_db = {}
    for dbfilename, dbconnection in db_connections.items():
        uids_to_row = {}
        for row in dbconnection.select(include_data=False):
            uids_to_row[row.uid] = row
        uids_for_each_db[dbfilename] = uids_to_row

    # loop over all rows of target DB and link to predefined links in links.json
    print(f"INFO: start linking to DB {target} ...")
    linkfilename = "links.json"
    for i, refrow in enumerate(link_db.select()):
        data = refrow.data
        if linkfilename in data:
            formatted_links = []
            uids_to_link_to = refrow.data[linkfilename]["uids"]
            for uid in uids_to_link_to:
                for dbfilename, uids_to_row in uids_for_each_db.items():
                    metadata = db_connections[dbfilename].metadata
                    row = uids_to_row.get(uid, None)
                    if not row:
                        continue
                    title = metadata["title"]
                    link_name_pattern = metadata["link_name"]
                    link_url_pattern = metadata["link_url"]
                    if row:
                        name = link_name_pattern.format(row=row, metadata=metadata)
                        url = link_url_pattern.format(row=row, metadata=metadata)
                        formatted_links.append((name, url, title))
            if formatted_links:
                data["links"] = formatted_links
                link_db.update(refrow.id, data={"links": data["links"]})
                print(
                    f"INFO: append links to row {i} ({refrow.formula}). "
                    f"Number of created links for this row: {len(formatted_links)}."
                )

    print(
        "===================================================="
        "===================================================="
    )
    print(f"INFO: finished linking to DB {target}!")

    return Result.fromdata(target_db=target, link_dbs=dbs)


def check_metadata(db, dbname, target):
    """Check metadata of a given database.

    Evaluate whether it is in accordance with the standard format needed
    for the crosslinks recipe.
    """
    metadata = db.metadata
    if target:
        print(f"INFO: check metadata of target DB ({dbname}) ...")
    elif not target:
        print(f"INFO: check metadata of linkage DB ({dbname}) ...")

    if "link_name" in metadata and "link_url" in metadata and "title" in metadata:
        pass
    else:
        raise KeyError(
            f"Metadata of DB ({dbname}) is not in "
            "accordance with the standard format needed "
            "for asr.database.crosslinks! "
            "The following keys "
            'are needed: "title", "link_name", "link_url"!'
        )

    return None
