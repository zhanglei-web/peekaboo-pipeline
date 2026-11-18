"""Convert stage-one proposals into stable memory units without inventing coverage."""
from copy import deepcopy


def initialize(catalog):
    if catalog.get("schema_version") != "online.candidate_catalog.v0.1":
        raise ValueError("Expected online.candidate_catalog.v0.1")
    episode_id = catalog["episode_id"]
    parents = {p["parent_id"]: p for p in catalog["parents"]}
    if len(parents) != len(catalog["parents"]):
        raise ValueError("Duplicate parent ID")
    units = {}
    for region in catalog["regions"]:
        uid = region["candidate_id"]
        if uid in units or region["parent_id"] not in parents:
            raise ValueError("Duplicate candidate ID or missing parent")
        anchor = region["identity_anchor"]
        kind = region["region_type"]
        if kind == "occluder_footprint" and anchor != "original_footprint":
            raise ValueError("Footprint must remain anchored to its original position")
        units[uid] = {
            "search_unit_id": uid,
            "candidate_id": uid,
            "parent_id": region["parent_id"],
            "macro_id": region.get("macro_id", catalog.get("macro_id", f"{episode_id}:scene")),
            "name": region["name"],
            "region_type": kind,
            "identity_anchor": anchor,
            "boundary_description": region["boundary_description"],
            "catalog_version": catalog["catalog_version"],
            "source": "stage1_candidate",
            "confirmed": False,
            "searched": False,
            "visit_count": 0,
            "last_q": None,
            "Q": None,
            "history": [],
        }
    return {
        "schema_version": "spatial.memory.v1",
        "episode_id": episode_id,
        "catalog_version": catalog["catalog_version"],
        "catalog_mode": catalog["mode"],
        "parents": deepcopy(catalog["parents"]),
        "units": units,
        "applied_events": [],
        "object_summary": {},
        "macro_summary": {},
    }
