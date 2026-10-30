"""Prepare a compact Quetta district MCDA grid from locally available data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.geometry import GeometryCollection, box, mapping
from shapely.ops import linemerge, unary_union


ROOT = Path(__file__).resolve().parent
WEB_DATA = ROOT / "web" / "data"
INPUT_DATA = ROOT / "data"
DATA_ROOT = Path(r"C:\webgis-data")
LAB_HOSPITALS = Path(r"C:\webgis-lab\quetta-dashboard\data\hospitals.geojson")
AREA_CRS = "EPSG:32642"
OUTPUT_CRS = "EPSG:4326"
CELL_SIZE_M = 100
ROAD_TYPES = ("motorway", "trunk", "primary", "secondary", "tertiary", "other")
DEFAULT_ROADS = DATA_ROOT / "qta_shp" / "roads.shp"
DEFAULT_WEIGHTS = {"pop": 30, "hosp": 25, "road": 20, "major": 15, "slope": 5, "hazard": 5}
CRITERIA = {
    "pop": ("Population", "Higher population", "People per cell"),
    "hosp": ("Hospital access", "Farther from existing hospitals", "Distance (m)"),
    "road": ("Road access", "Closer to roads", "Distance (m)"),
    "major": ("Primary / secondary road access", "Closer to primary or secondary roads", "Distance (m)"),
    "slope": ("Terrain slope", "Flatter terrain", "Slope (degrees)"),
    "hazard": ("River separation", "Farther from rivers", "Distance (m)"),
}
POPULATION_URL = "https://hub.worldpop.org/geodata/summary?id=24777"
DEM_URL = "https://earthexplorer.usgs.gov/"


def log_inventory(root: Path, report_path: Path) -> None:
    """Print every source file and the schema of each readable vector layer."""
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []
    print(f"\nData inventory: {root} ({len(files)} files)")
    for path in files:
        print(f"  FILE {path.relative_to(root)}")

    vector_sources: list[tuple[Path, str, str]] = []
    for path in files:
        if path.suffix.casefold() in {".geojson", ".json", ".shp", ".gpkg"}:
            vector_sources.append((path, str(path), path.stem))
        elif path.suffix.casefold() == ".zip":
            try:
                with zipfile.ZipFile(path) as archive:
                    for member in archive.namelist():
                        if member.casefold().endswith(".shp"):
                            vector_sources.append(
                                (path, f"zip://{path}!{member}", Path(member).stem)
                            )
            except zipfile.BadZipFile as error:
                logging.warning("Could not inspect archive %s: %s", path, error)
    records: list[dict[str, Any]] = []
    inspected_files: set[Path] = set()
    for path, source, layer_name in vector_sources:
        try:
            frame = gpd.read_file(source)
            geometry_types = ", ".join(sorted(frame.geometry.geom_type.dropna().unique()))
            record = {
                "file": str(path.relative_to(root)),
                "layer": layer_name,
                "geometry": geometry_types or "None",
                "crs": str(frame.crs) if frame.crs else "None",
                "features": len(frame),
                "columns": ", ".join(map(str, frame.columns)),
            }
            records.append(record)
            inspected_files.add(path)
            print(
                f"  LAYER {record['layer']} | geometry={record['geometry']} | "
                f"CRS={record['crs']} | features={record['features']} | "
                f"columns=[{record['columns']}]"
            )
        except Exception as error:
            logging.warning("Could not inspect vector %s: %s", path, error)
    for path in files:
        if path not in inspected_files:
            records.append(
                {
                    "file": str(path.relative_to(root)),
                    "layer": "",
                    "geometry": "",
                    "crs": "",
                    "features": "",
                    "columns": "",
                }
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["file", "layer", "geometry", "crs", "features", "columns"]
        )
        writer.writeheader()
        writer.writerows(records)


def read_vector(path: Path | None, label: str) -> gpd.GeoDataFrame:
    if path is None:
        return gpd.GeoDataFrame(geometry=[], crs=OUTPUT_CRS)
    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError(f"{label} source has no CRS: {path}")
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    return frame.to_crs(AREA_CRS)


def find_boundary(path: Path | None) -> tuple[gpd.GeoDataFrame, Path]:
    candidates = [path] if path else [
        DATA_ROOT / "from_repo" / "pak_admin2.shp",
        DATA_ROOT / "from_repo" / "district_c.geojson",
        DATA_ROOT / "from_repo" / "districts_A.geojson",
        ROOT / "data" / "quetta_district.geojson",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        frame = gpd.read_file(candidate)
        if frame.crs is None:
            raise ValueError(f"Boundary source has no CRS: {candidate}")
        frame = frame.to_crs(AREA_CRS)
        fields = [
            field
            for field in frame.columns
            if field != frame.geometry.name
            and any(token in str(field).casefold() for token in ("district", "adm2", "name"))
        ]
        for field in fields:
            matches = frame[
                frame[field].astype("string").str.contains("quetta", case=False, na=False)
            ]
            if not matches.empty:
                geometry = unary_union(matches.geometry)
                result = gpd.GeoDataFrame(
                    {"district": ["Quetta"]}, geometry=[geometry], crs=AREA_CRS
                )
                return result, candidate
    raise FileNotFoundError(
        "No Quetta district boundary found. Expected C:\\webgis-data\\from_repo\\pak_admin2.shp "
        "or pass --boundary."
    )


def find_urban_features(district: Any, center: Any) -> list[Any]:
    """Return local building/urban polygons near the densest hospital cluster."""
    urban_features: list[Any] = []
    urban_area = center.buffer(5_000)
    for root in (DATA_ROOT, INPUT_DATA):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {
                ".geojson", ".json", ".shp", ".gpkg"
            }:
                continue
            try:
                frame = gpd.read_file(path)
                if frame.crs is None or frame.empty:
                    continue
                building_field = field_name(frame, ("building",))
                landuse_field = field_name(frame, ("landuse", "land_use"))
                urban_field = field_name(frame, ("urban", "urban_area"))
                if not any((building_field, landuse_field, urban_field)):
                    continue
                frame = frame.to_crs(AREA_CRS)
                polygonal = frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                selected = polygonal.copy()
                if building_field:
                    values = frame[building_field].astype("string").str.strip().str.casefold()
                    selected &= values.notna() & ~values.isin({"", "no", "false", "0", "nan"})
                elif landuse_field:
                    values = frame[landuse_field].astype("string").str.strip().str.casefold()
                    selected &= values.isin({
                        "residential", "commercial", "retail", "industrial",
                        "construction", "mixed", "urban",
                    })
                else:
                    values = frame[urban_field].astype("string").str.strip().str.casefold()
                    selected &= values.isin({"yes", "true", "1", "urban"})
                features = frame.loc[selected].geometry
                features = features.loc[features.intersects(urban_area)]
                urban_features.extend(
                    geometry.intersection(district)
                    for geometry in features
                    if geometry.intersects(district)
                )
            except Exception as error:
                logging.debug("Skipping urban-feature candidate %s: %s", path, error)
    return [geometry for geometry in urban_features if not geometry.is_empty]


def find_city_boundary(
    district: gpd.GeoDataFrame, hospital_path: Path | None
) -> tuple[gpd.GeoDataFrame, str]:
    hospitals = choose_points(hospital_path, district, "hospitals for urban-core boundary")
    if hospitals.empty:
        raise FileNotFoundError(
            "Cannot build the Quetta urban core without local hospital points."
        )
    coordinates = np.array([(point.x, point.y) for point in hospitals.geometry])
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2))
    center_index = int((distances <= 2_500).sum(axis=1).argmax())
    center = hospitals.geometry.iloc[center_index]
    core_hospitals = hospitals.loc[distances[center_index] <= 3_500]
    urban_features = find_urban_features(district.geometry.iloc[0], center)
    source_geometries = [*core_hospitals.geometry, *urban_features]
    if not urban_features:
        logging.warning(
            "No local building or urban-landuse polygons found; using the dense central "
            "%s-hospital cluster.",
            len(core_hospitals),
        )
    geometry = unary_union(source_geometries).convex_hull.buffer(1_500)
    geometry = geometry.intersection(district.geometry.iloc[0])
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("The generated Quetta urban-core boundary is empty.")
    city = gpd.GeoDataFrame({"name": ["Quetta urban core"]}, geometry=[geometry], crs=AREA_CRS)
    return city, "dense local hospital cluster, with nearby urban polygons where available"


def discover_layer(
    explicit: Path | None,
    roots: list[Path],
    required_fields: set[str],
    accepted_geometry: set[str],
) -> Path | None:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"Input does not exist: {explicit}")
        return explicit
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {
                ".geojson", ".json", ".shp", ".gpkg"
            }:
                continue
            try:
                frame = gpd.read_file(path, rows=1) if path.suffix.casefold() == ".gpkg" else gpd.read_file(path)
                columns = {str(column).casefold() for column in frame.columns}
                if path.stem.casefold() in {"pak_adminlines", "pak_adminlines_em"}:
                    continue
                if required_fields.intersection(columns) and frame.geometry.geom_type.isin(
                    accepted_geometry
                ).any():
                    return path
            except Exception:
                continue
    return None


def field_name(frame: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> str | None:
    fields = {str(column).casefold(): column for column in frame.columns}
    return next((fields[name] for name in candidates if name in fields), None)


def normalize_road_type(value: Any) -> str | None:
    road_type = re.sub(r"\s+", "_", str(value).strip().casefold())
    if road_type in {"", "<na>", "nan", "none"}:
        return None
    road_type = re.sub(r"(?:_link|-link| link)$", "", road_type)
    return road_type if road_type in ROAD_TYPES[:-1] else "other"


def choose_points(
    path: Path | None, boundary: gpd.GeoDataFrame, label: str
) -> gpd.GeoDataFrame:
    frame = read_vector(path, label)
    if frame.empty:
        return frame
    frame = frame.loc[frame.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if frame.empty:
        return frame
    district = boundary.geometry.iloc[0]
    return frame.loc[frame.geometry.intersects(district)].copy()


def choose_network(
    path: Path | None,
    boundary: gpd.GeoDataFrame,
    *,
    kind: str,
) -> gpd.GeoDataFrame:
    frame = read_vector(path, kind)
    if frame.empty:
        return frame
    frame = frame.loc[
        frame.geometry.geom_type.isin(["LineString", "MultiLineString", "Polygon", "MultiPolygon"])
    ].copy()
    if frame.empty:
        return frame
    district = boundary.geometry.iloc[0]
    clip_geometry = district.buffer(-1) if kind == "roads" else district
    frame = frame.loc[frame.geometry.intersects(clip_geometry)].copy()
    frame.geometry = frame.geometry.intersection(clip_geometry)
    frame = frame.loc[~frame.geometry.is_empty].copy()

    if kind == "roads":
        highway_field = field_name(
            frame, ("highway", "fclass", "road_type", "type", "roadtype", "class")
        )
        if highway_field is None:
            raise ValueError(f"Road source {path} has no highway/fclass/type field.")
        name_field = field_name(frame, ("name", "name:en", "ref"))
        frame["road_type"] = frame[highway_field].map(normalize_road_type)
        frame["road_name"] = (
            frame[name_field].astype("string").str.strip().fillna("")
            if name_field
            else ""
        )
        frame["road_name"] = frame["road_name"].replace({"<NA>": "", "nan": ""})
        frame = frame.loc[frame["road_type"].isin(ROAD_TYPES)].copy()
        if frame.empty:
            return frame
        merged: list[dict[str, Any]] = []
        for (name, road_type), group in frame.groupby(
            ["road_name", "road_type"], dropna=False, sort=False
        ):
            geometry = unary_union(group.geometry)
            if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
                geometry = linemerge(geometry)
            if geometry.geom_type == "GeometryCollection":
                geometry = GeometryCollection(
                    [part for part in geometry.geoms if part.geom_type in {"LineString", "MultiLineString"}]
                )
            geometry = geometry.simplify(2, preserve_topology=True).intersection(district)
            if geometry.is_empty:
                continue
            merged.append(
                {"name": str(name) if name else "", "highway": str(road_type), "geometry": geometry}
            )
        return gpd.GeoDataFrame(merged, geometry="geometry", crs=AREA_CRS)

    kind_field = field_name(frame, ("waterway", "natural", "fclass", "type", "water"))
    if kind_field:
        values = frame[kind_field].astype("string").str.casefold()
        frame = frame.loc[
            values.str.contains("river|stream|water|canal|drain", na=False)
        ].copy()
    return frame


def create_grid(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    city = boundary.geometry.iloc[0]
    min_x, min_y, max_x, max_y = city.bounds
    start_x = math.floor(min_x / CELL_SIZE_M) * CELL_SIZE_M
    start_y = math.floor(min_y / CELL_SIZE_M) * CELL_SIZE_M
    cells: list[dict[str, Any]] = []
    cell_id = 1
    for x in np.arange(start_x, max_x, CELL_SIZE_M):
        for y in np.arange(start_y, max_y, CELL_SIZE_M):
            clipped = box(x, y, x + CELL_SIZE_M, y + CELL_SIZE_M).intersection(city)
            if clipped.is_empty or clipped.area < 1:
                continue
            cells.append({"cell_id": cell_id, "area_km2": clipped.area / 1_000_000, "geometry": clipped})
            cell_id += 1
    return gpd.GeoDataFrame(cells, geometry="geometry", crs=AREA_CRS)


def quantile_scores(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    valid = np.isfinite(values)
    scores = np.full(values.shape, np.nan, dtype=float)
    if not valid.any():
        return scores
    oriented = values[valid] if higher_is_better else -values[valid]
    ranks = rankdata(oriented, method="average")
    scores[valid] = np.clip(np.ceil(ranks / len(ranks) * 5), 1, 5)
    return scores


def distances_to_network(
    centroids: gpd.GeoSeries, network: gpd.GeoDataFrame
) -> np.ndarray:
    if network.empty:
        return np.full(len(centroids), np.nan)
    merged = unary_union(network.geometry)
    return np.fromiter((point.distance(merged) for point in centroids), dtype=float, count=len(centroids))


def zonal_means(
    grid: gpd.GeoDataFrame,
    raster_path: Path,
    *,
    stats: tuple[str, ...] = ("mean",),
) -> np.ndarray:
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        zones = grid.to_crs(source.crs)
        result = zonal_stats(
            zones.geometry,
            raster_path,
            stats=list(stats),
            nodata=source.nodata,
            all_touched=False,
        )
    return np.array([item.get("mean", np.nan) for item in result], dtype=float)


def slope_raster(dem_path: Path, district: gpd.GeoDataFrame) -> tuple[np.ndarray, Any, dict[str, Any]]:
    district_geom = district.geometry.iloc[0]
    with rasterio.open(dem_path) as source:
        if source.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        bounds = transform_bounds(
            AREA_CRS, AREA_CRS, *district.total_bounds, densify_pts=21
        )
        projected_bounds = transform_bounds(
            source.crs, AREA_CRS, *source.bounds, densify_pts=21
        )
        left = max(projected_bounds[0], district_geom.bounds[0] - 1000)
        bottom = max(projected_bounds[1], district_geom.bounds[1] - 1000)
        right = min(projected_bounds[2], district_geom.bounds[2] + 1000)
        top = min(projected_bounds[3], district_geom.bounds[3] + 1000)
        if not left < right or not bottom < top:
            raise ValueError(f"DEM does not cover the Quetta boundary: {dem_path}")

        native_resolution = max(abs(value) for value in source.res)
        if source.crs.is_geographic:
            native_resolution *= 111_000
        resolution = min(200, max(15, native_resolution))
        width = max(1, math.ceil((right - left) / resolution))
        height = max(1, math.ceil((top - bottom) / resolution))
        transform = from_origin(left, top, resolution, resolution)
        nodata = source.nodata if source.nodata is not None else -9999
        elevation = np.full((height, width), nodata, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=elevation,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=transform,
            dst_crs=AREA_CRS,
            dst_nodata=nodata,
            resampling=Resampling.bilinear,
        )
        invalid = ~np.isfinite(elevation) | (elevation == nodata)
        if invalid.all():
            raise ValueError(f"DEM has no valid coverage over Quetta: {dem_path}")
        elevation[invalid] = np.nan
        filled = fillnodata(
            np.nan_to_num(elevation, nan=-9999),
            mask=(~invalid).astype(np.uint8),
            max_search_distance=100,
            smoothing_iterations=0,
        )
        filled[invalid] = np.nan
        dy, dx = np.gradient(np.nan_to_num(filled, nan=float(np.nanmedian(filled))))
        slope = np.degrees(np.arctan(np.hypot(dx / resolution, dy / resolution))).astype(np.float32)
        slope[invalid] = -9999
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": AREA_CRS,
            "transform": transform,
            "nodata": -9999,
            "compress": "deflate",
        }
    return slope, transform, profile


def write_slope_raster(
    raster_path: Path, array: np.ndarray, profile: dict[str, Any]
) -> None:
    with rasterio.open(raster_path, "w", **profile) as target:
        target.write(array, 1)


def round_coordinates(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: round_coordinates(part) for key, part in value.items()}
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], (float, int, np.floating, np.integer)):
            return [round(float(part), 5) if isinstance(part, (float, int, np.floating, np.integer)) else part for part in value]
        return [round_coordinates(part) for part in value]
    return value


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if not isinstance(value, (str, list, dict)) else value


def write_geojson(
    frame: gpd.GeoDataFrame,
    path: Path,
    *,
    simplify_m: float = 0,
    columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    working = frame.to_crs(AREA_CRS)
    output = working.to_crs(OUTPUT_CRS)
    features = []
    for _, row in output.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        if simplify_m:
            projected = gpd.GeoSeries([geometry], crs=OUTPUT_CRS).to_crs(AREA_CRS).iloc[0]
            geometry = gpd.GeoSeries(
                [projected.simplify(simplify_m, preserve_topology=True)], crs=AREA_CRS
            ).to_crs(OUTPUT_CRS).iloc[0]
        props = {
            str(column): json_value(row[column])
            for column in (columns or [c for c in output.columns if c != output.geometry.name])
        }
        features.append(
            {
                "type": "Feature",
                "geometry": round_coordinates(mapping(geometry)),
                "properties": props,
            }
        )
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"type": "FeatureCollection", "features": features},
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--hospitals", type=Path)
    parser.add_argument("--roads", type=Path)
    parser.add_argument("--rivers", type=Path)
    parser.add_argument("--water", type=Path)
    parser.add_argument("--population", type=Path, default=INPUT_DATA / "pak_ppp.tif")
    parser.add_argument("--dem", type=Path, default=INPUT_DATA / "srtm.tif")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    WEB_DATA.mkdir(parents=True, exist_ok=True)
    log_inventory(DATA_ROOT, ROOT / "data_inventory.csv")
    boundary, boundary_path = find_boundary(args.boundary)
    logging.info("Boundary: %s (Quetta district)", boundary_path)
    hospital_path = args.hospitals
    if hospital_path is None and LAB_HOSPITALS.exists():
        hospital_path = LAB_HOSPITALS
    city, city_path = find_city_boundary(boundary, hospital_path)
    city_area_km2 = city.geometry.iloc[0].area / 1_000_000
    logging.info(
        "Urban core: %.2f km2; bbox [%.5f, %.5f, %.5f, %.5f] (EPSG:4326)",
        city_area_km2,
        *city.to_crs(OUTPUT_CRS).total_bounds,
    )
    city_interior = city.copy()
    city_interior.geometry = city.geometry.buffer(-1)
    roots = [DATA_ROOT, INPUT_DATA]
    hospitals = choose_points(hospital_path, boundary, "hospitals")
    city_hospitals = choose_points(hospital_path, city_interior, "city hospitals")

    roads_path = args.roads or (
        DEFAULT_ROADS if DEFAULT_ROADS.exists() else discover_layer(
            None,
            roots,
            {"highway", "fclass", "road_type", "roadtype", "type", "class"},
            {"LineString", "MultiLineString"},
        )
    )
    roads = choose_network(roads_path, city_interior, kind="roads") if roads_path else gpd.GeoDataFrame(
        columns=["name", "highway", "geometry"], geometry="geometry", crs=AREA_CRS
    )

    rivers_path = args.rivers
    if rivers_path is None:
        rivers_path = discover_layer(
            None, roots, {"waterway"}, {"LineString", "MultiLineString"}
        )
    rivers = choose_network(rivers_path, city_interior, kind="rivers") if rivers_path else gpd.GeoDataFrame(
        geometry=[], crs=AREA_CRS
    )

    water_path = args.water
    if water_path is None:
        water_path = discover_layer(
            None, roots, {"natural", "water"}, {"Polygon", "MultiPolygon"}
        )
    water = choose_network(water_path, city_interior, kind="water") if water_path else gpd.GeoDataFrame(
        geometry=[], crs=AREA_CRS
    )

    if len(hospitals):
        logging.info("Hospitals: %s (%s)", hospital_path, len(hospitals))
    else:
        logging.warning("No local hospital points found; hospital-access criterion will be skipped.")
    logging.info("Hospitals inside the urban core: %s", len(city_hospitals))
    if roads.empty:
        logging.warning("No road features were found inside the Quetta urban-core boundary.")
    else:
        logging.info("City-clipped roads: %s (%s merged segments)", roads_path, len(roads))
        logging.info(
            "First five city road properties: %s",
            roads[["name", "highway"]].head(5).to_dict(orient="records"),
        )
    if rivers.empty:
        logging.warning("No river layer found; hazard criterion will be skipped and weights redistributed.")
    if water.empty:
        logging.info("No water polygons supplied; water-area exclusions cannot be applied.")

    grid = create_grid(city_interior)
    centroids = grid.geometry.centroid
    criterion_available: dict[str, bool] = {}
    raw: dict[str, np.ndarray] = {
        "pop": np.full(len(grid), np.nan),
        "hosp": np.full(len(grid), np.nan),
        "road": np.full(len(grid), np.nan),
        "major": np.full(len(grid), np.nan),
        "slope": np.full(len(grid), np.nan),
        "hazard": np.full(len(grid), np.nan),
    }

    for key in raw:
        grid[f"raw_{key}"] = np.nan
        grid[f"score_{key}"] = np.nan

    if args.population.exists():
        raw["pop"] = zonal_means(grid, args.population)
        criterion_available["pop"] = np.isfinite(raw["pop"]).any()
    else:
        criterion_available["pop"] = False
        logging.warning(
            "Population raster missing: %s. Download: %s",
            args.population,
            POPULATION_URL,
        )
    if not city_hospitals.empty:
        hospital_tree = cKDTree(np.array([(geom.x, geom.y) for geom in city_hospitals.geometry]))
        raw["hosp"] = hospital_tree.query(
            np.column_stack((centroids.x.to_numpy(), centroids.y.to_numpy())), k=1
        )[0]
        criterion_available["hosp"] = True
    else:
        criterion_available["hosp"] = False
    if not roads.empty:
        raw["road"] = distances_to_network(centroids, roads)
        criterion_available["road"] = True
    else:
        criterion_available["road"] = False
    major_roads = roads.loc[roads["highway"].isin({"primary", "secondary"})] if not roads.empty else roads
    if not major_roads.empty:
        raw["major"] = distances_to_network(centroids, major_roads)
        criterion_available["major"] = True
    else:
        criterion_available["major"] = False
    if args.dem.exists():
        slope_values, _, profile = slope_raster(args.dem, city_interior)
        with tempfile.TemporaryDirectory(prefix="quetta-slope-") as directory:
            slope_path = Path(directory) / "slope.tif"
            write_slope_raster(slope_path, slope_values, profile)
            raw["slope"] = zonal_means(grid, slope_path)
        criterion_available["slope"] = np.isfinite(raw["slope"]).any()
    else:
        criterion_available["slope"] = False
        logging.warning("DEM missing: %s. Download: %s", args.dem, DEM_URL)
    if not rivers.empty:
        raw["hazard"] = distances_to_network(centroids, rivers)
        criterion_available["hazard"] = True
    else:
        criterion_available["hazard"] = False

    for key, values in raw.items():
        grid[f"raw_{key}"] = values
        if criterion_available.get(key, False):
            grid[f"score_{key}"] = quantile_scores(
                values, higher_is_better=key in {"pop", "hosp", "hazard"}
            )

    excluded = np.zeros(len(grid), dtype=bool)
    if criterion_available.get("slope", False):
        excluded |= raw["slope"] > 15
    if not water.empty:
        water_union = unary_union(water.geometry)
        excluded |= np.fromiter(
            (geometry.intersects(water_union) for geometry in grid.geometry),
            dtype=bool,
            count=len(grid),
        )
    if not city_hospitals.empty:
        hospital_cells = gpd.sjoin(
            city_hospitals[["geometry"]],
            grid[["cell_id", "geometry"]],
            how="inner",
            predicate="intersects",
        )
        excluded_ids = set(hospital_cells["cell_id"].dropna().astype(int))
        excluded |= grid["cell_id"].isin(excluded_ids).to_numpy()
    grid["excluded"] = excluded.astype(int)
    grid["suitability"] = np.nan
    grid["suit_class"] = 0
    city_geometry = city_interior.geometry.iloc[0]
    city_cells = grid.loc[grid.geometry.intersects(city_geometry)].copy()
    city_cells["area_km2"] = city_cells.geometry.area / 1_000_000

    active = [key for key, available in criterion_available.items() if available]
    weight_total = sum(DEFAULT_WEIGHTS[key] for key in active)
    normalized_weights = {
        key: DEFAULT_WEIGHTS[key] * 100 / weight_total for key in active
    } if weight_total else {}
    metadata = {
        "district": "Quetta",
        "crs": OUTPUT_CRS,
        "analysis_crs": AREA_CRS,
        "cell_size_m": CELL_SIZE_M,
        "city_area_km2": round(city_area_km2, 3),
        "city_bounds_wgs84": [round(float(value), 5) for value in city.to_crs(OUTPUT_CRS).total_bounds],
        "criteria": [
            {
                "key": key,
                "label": CRITERIA[key][0],
                "preference": CRITERIA[key][1],
                "units": CRITERIA[key][2],
                "available": bool(criterion_available.get(key, False)),
                "default_weight": round(normalized_weights.get(key, 0), 4),
            }
            for key in CRITERIA
        ],
        "active_criteria": active,
        "default_weights": normalized_weights,
        "sources": {
            "boundary": str(boundary_path),
            "city_boundary": str(city_path),
            "hospitals": str(hospital_path) if hospital_path else None,
            "roads": str(roads_path) if roads_path else None,
            "rivers": str(rivers_path) if rivers_path else None,
            "population": str(args.population) if args.population.exists() else None,
            "dem": str(args.dem) if args.dem.exists() else None,
        },
        "counts": {
            "grid_cells": len(city_cells),
            "hospitals": len(city_hospitals),
            "roads": len(roads),
            "major_roads": len(major_roads),
            "rivers": len(rivers),
            "excluded_cells": int(city_cells["excluded"].sum()),
        },
        "road_types": {
            road_type: {
                "count": int((roads["highway"] == road_type).sum()) if not roads.empty else 0,
                "length_km": round(
                    float(roads.loc[roads["highway"] == road_type].geometry.length.sum() / 1000),
                    2,
                ) if not roads.empty else 0,
            }
            for road_type in ROAD_TYPES
        },
    }

    grid_columns = [
        "cell_id", "area_km2", "excluded",
        *[f"raw_{key}" for key in CRITERIA],
        *[f"score_{key}" for key in CRITERIA],
    ]
    write_geojson(city_cells[grid_columns + ["geometry"]], WEB_DATA / "suitability_grid.geojson")
    write_geojson(boundary[["district", "geometry"]], WEB_DATA / "boundary.geojson", simplify_m=15)
    write_geojson(city[["name", "geometry"]], WEB_DATA / "quetta_city.geojson")
    write_geojson(city_hospitals, WEB_DATA / "hospitals.geojson")
    road_export = roads.rename(columns={"road_name": "name"}) if "road_name" in roads else roads
    road_columns = [column for column in ("name", "highway", "geometry") if column in road_export]
    write_geojson(road_export[road_columns], WEB_DATA / "roads_city.geojson")
    legacy_roads = WEB_DATA / "roads.geojson"
    if legacy_roads.exists():
        legacy_roads.unlink()
        logging.info("Removed stale uncropped roads layer: %s", legacy_roads)
    (WEB_DATA / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info(
        "Wrote %s 100 m city cells covering %.2f km2; available criteria: %s; default weights rebalanced to %s.",
        len(city_cells),
        city_cells["area_km2"].sum(),
        ", ".join(active) if active else "none",
        normalized_weights,
    )
    logging.info("Road type counts inside the Quetta urban core (exported features):")
    for road_type in ROAD_TYPES:
        stats = metadata["road_types"][road_type]
        logging.info(
            "  %s: %s features, %.2f km",
            "residential/other" if road_type == "other" else road_type,
            stats["count"],
            stats["length_km"],
        )
    for path in sorted(WEB_DATA.glob("*")):
        logging.info("Output: %s (%.2f MB)", path, path.stat().st_size / 1_000_000)


if __name__ == "__main__":
    main()
