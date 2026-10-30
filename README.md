# Quetta Hospital Site Suitability

Static Leaflet MCDA dashboard and local-data preparation script for the Quetta
urban core. The workflow never downloads OSM; it uses the supplied district
layers and locally available roads, rivers, rasters, and hospital points.

## Prepare and serve

```powershell
Set-Location C:\quetta_mcda
python -m pip install geopandas rasterio shapely pyproj numpy scipy rasterstats
python prep.py
Set-Location C:\quetta_mcda\web
python -m http.server 8000
```

Open <http://localhost:8000>. Leaflet, Chart.js, leaflet-textpath, fonts, and
the selected basemaps are loaded from CDNs, so those interface resources need
an internet connection. CARTO is included as a selectable basemap, but it
returned API-key-required tiles in this environment; the darkened, keyless OSM
layer is selected by default instead.

## Publish with GitHub Pages

Pushing to `main` automatically publishes the dashboard from `web/` using
GitHub Actions. After the first deployment completes, open
<https://amjad0978.github.io/MCDA-Quetta-City-Hospitals-/>.

The district boundary is
`C:\webgis-data\from_repo\pak_admin2.shp`. The urban-core extent is built from
the dense cluster of existing hospitals, buffered by 1,500 m and clipped to
the district. Local building/urban-landuse polygons are included when found;
none are present in the supplied `C:\webgis-data` snapshot. If present, the
existing hospital snapshot at
`C:\webgis-lab\quetta-dashboard\data\hospitals.geojson` is reused. Supply
optional inputs with `--hospitals`, `--roads`, `--rivers`, `--water`,
`--population`, and `--dem`. The default roads source is
`C:\webgis-data\qta_shp\roads.shp`. Its `fclass` and `name` attributes are
normalized, clipped to the urban-core polygon, simplified by about two metres,
and written as `web\data\roads_city.geojson`. The 100 m EPSG:32642 grid and
nearest-hospital, nearest-road, and nearest-primary/secondary-road distances
are calculated only inside the core. The dashboard displays only urban-core
roads, hospitals, and grid cells. Put population and SRTM rasters in
`data\pak_ppp.tif` and `data\srtm.tif`, respectively, or pass their paths
explicitly.

The script prints the complete `C:\webgis-data` file inventory and vector
schemas, and writes the machine-readable layer report to `data_inventory.csv`.
Generated map files, including `quetta_city.geojson`, are written to
`web\data\`. Criteria without local data are
disabled and their configured weight share is redistributed across available
criteria. WLC and TOPSIS use min-max criterion values over eligible urban-core
cells, and the suitability legend and area chart use five city-wide quantile
classes. The road summary prints normalized road-type feature counts and total
urban-core lengths. River, population, and DEM data remain optional and are not
downloaded.

Population data: <https://hub.worldpop.org/geodata/summary?id=24777>  
SRTM elevation data: <https://earthexplorer.usgs.gov/>

Suitability is a screening aid, not a site approval. Confirm candidate parcels,
hazards, access, utilities, land tenure, and facility coverage with authoritative
local sources before making planning decisions.
