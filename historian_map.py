import os

import pandas as pd
import plotly.express as px
import streamlit as st
from influxdb_client import InfluxDBClient


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return default


def get_influx_credentials():
    load_env_file(".env")
    url = env_first("INFLUXDB_URL", "INFLUX_URL")
    token = env_first("INFLUXDB_TOKEN", "INFLUX_TOKEN")
    org = env_first("INFLUXDB_ORG", "INFLUX_ORG")
    bucket = env_first("INFLUXDB_BUCKET", "INFLUX_BUCKET")
    if not url or not token or not org:
        missing = [k for k, v in {
            "INFLUXDB_URL/INFLUX_URL": url,
            "INFLUXDB_TOKEN/INFLUX_TOKEN": token,
            "INFLUXDB_ORG/INFLUX_ORG": org,
        }.items() if not v]
        raise RuntimeError("Missing InfluxDB credentials: " + ", ".join(missing))
    return url, token, org, bucket


def format_flux_list(values, column):
    return " or ".join([f'r.{column} == "{val}"' for val in values])


def build_noaa_flux_query(bucket, start, stop, measurement, fields, tag_key, tag_values, use_pivot=False):
    field_filter = format_flux_list(fields, "_field")
    tag_filter_clause = ""
    if tag_values:
        tag_filter = format_flux_list(tag_values, tag_key)
        tag_filter_clause = f'\n  |> filter(fn: (r) => {tag_filter})'
    keep_cols = f'["_time", "_value", "_field", "{tag_key}"]'
    pivot_clause = ""
    if use_pivot:
        pivot_clause = f'\n  |> pivot(rowKey:["_time","{tag_key}"], columnKey: ["_field"], valueColumn: "_value")'
    return f"""
from(bucket: "{bucket}")
  |> range(start: time(v: "{start}"), stop: time(v: "{stop}"))
  |> filter(fn: (r) => r._measurement == "{measurement}")
  |> filter(fn: (r) => {field_filter})
  {tag_filter_clause}
  |> keep(columns: {keep_cols})
  {pivot_clause}
  |> sort(columns: ["_time"])
""".strip()


@st.cache_data(show_spinner=False, ttl=600)
def fetch_noaa_from_influx(bucket, start, stop, measurement, fields, tag_key, tag_values, use_pivot=False):
    url, token, org, _ = get_influx_credentials()
    flux = build_noaa_flux_query(bucket, start, stop, measurement, fields, tag_key, tag_values, use_pivot=use_pivot)
    client = InfluxDBClient(url=url, token=token, org=org)
    query_api = client.query_api()
    data = query_api.query_data_frame(flux)
    client.close()
    if isinstance(data, list):
        df = pd.concat(data, ignore_index=True)
    else:
        df = data
    return df


st.set_page_config(page_title="Historian — NOAA Sample Map", layout="wide")
st.title("Historian — NOAA Sample Map (InfluxDB)")
st.markdown(
    "This is a **pipeline validation** demo. NOAA sample data is **not** Black Sea data."
)

with st.expander("Load NOAA sample into InfluxDB (one-time)"):
    st.code(
        'import "influxdata/influxdb/sample"\n\n'
        'sample.data(set: "noaaWater")\n'
        '    |> to(bucket: "example-bucket")'
    )
    st.caption("Replace example-bucket with your target bucket.")

try:
    _, _, _, default_bucket = get_influx_credentials()
except Exception:
    default_bucket = ""

dataset_preset = st.selectbox(
    "Dataset preset",
    [
        "NOAA water level (noaaWater)",
        "NOAA temperature (average_temperature)",
        "Black Sea (Copernicus ingest)"
    ]
)

relocate_available = dataset_preset.startswith("NOAA")

if dataset_preset == "Black Sea (Copernicus ingest)":
    measurement = "blacksea_bgc"
    fields = ["value", "latitude", "longitude"]
    tag_key = "variable"
    variable_choice = st.selectbox("Variable", ["chl", "phyc", "zooc"])
    tag_values = [variable_choice]
    use_pivot = True
    now = pd.Timestamp.utcnow()
    default_start = (now - pd.Timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    default_stop = now.strftime("%Y-%m-%dT%H:%M:%SZ")
else:
    if dataset_preset == "NOAA temperature (average_temperature)":
        measurement = "average_temperature"
        fields = ["degrees"]
        tag_key = "location"
        tag_values = ["coyote_creek", "santa_monica"]
        use_pivot = False
    else:
        measurement = "h2o_feet"
        fields = ["water_level"]
        tag_key = "station_id"
        tag_values = ["9410840", "9414575"]
        use_pivot = False
    default_start = "2019-08-17T00:00:00Z"
    default_stop = "2019-09-17T22:00:00Z"

bucket = st.text_input("Influx bucket", value=default_bucket or "")
col_a, col_b = st.columns(2)
with col_a:
    start_time = st.text_input("Start (RFC3339)", value=default_start)
with col_b:
    stop_time = st.text_input("Stop (RFC3339)", value=default_stop)

st.markdown("**Dataset (hardcoded preset)**")
st.code(
    f"measurement: {measurement}\n"
    f"field: {', '.join(fields)}\n"
    f"{tag_key}: {', '.join(tag_values)}"
)
if dataset_preset == "Black Sea (Copernicus ingest)":
    st.caption("Source: Copernicus Marine (ingested into Influx).")

station_state_key = f"noaa_station_locations_{tag_key}"
if station_state_key not in st.session_state:
    st.session_state[station_state_key] = None

if st.button("Fetch NOAA data from InfluxDB"):
    try:
        if not bucket:
            st.error("Bucket is required.")
        else:
            with st.spinner("Querying InfluxDB..."):
                df_noaa = fetch_noaa_from_influx(
                    bucket=bucket,
                    start=start_time,
                    stop=stop_time,
                    measurement=measurement,
                    fields=fields,
                    tag_key=tag_key,
                    tag_values=tag_values,
                    use_pivot=use_pivot
                )
            if df_noaa is None or df_noaa.empty:
                st.warning("No data returned.")
            else:
                st.success(f"Fetched {len(df_noaa)} records.")
                st.dataframe(df_noaa.head(50), use_container_width=True)

                if dataset_preset == "Black Sea (Copernicus ingest)":
                    if not {"latitude", "longitude", "value"}.issubset(df_noaa.columns):
                        st.warning("Missing latitude/longitude/value fields. Re-run ingestion.")
                        st.stop()

                    plot_df = df_noaa.dropna(subset=["latitude", "longitude", "value"]).copy()
                    if len(plot_df) > 20000:
                        plot_df = plot_df.sample(20000)

                    vmin = plot_df["value"].quantile(0.05)
                    vmax = plot_df["value"].quantile(0.95)
                    fig = px.density_mapbox(
                        plot_df,
                        lat="latitude",
                        lon="longitude",
                        z="value",
                        radius=8,
                        zoom=5,
                        center=dict(lat=43, lon=35),
                        mapbox_style="carto-positron",
                        color_continuous_scale="Viridis",
                        range_color=(vmin, vmax),
                        title=f"Black Sea — {tag_values[0]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("**Time-series preview (mean over grid)**")
                    ts_df = df_noaa.groupby("_time", as_index=False)["value"].mean()
                    ts_fig = px.line(ts_df, x="_time", y="value", title=f"{tag_values[0]} — mean")
                    st.plotly_chart(ts_fig, use_container_width=True)
                else:
                    station_values = []
                    if tag_key in df_noaa.columns:
                        station_values = sorted(df_noaa[tag_key].dropna().unique().tolist())
                    if not station_values:
                        station_values = tag_values

                    if (
                        st.session_state[station_state_key] is None
                        or set(st.session_state[station_state_key][tag_key]) != set(station_values)
                    ):
                        st.session_state[station_state_key] = pd.DataFrame(
                            [{tag_key: sid, "name": f"{tag_key}: {sid}", "lat": None, "lon": None} for sid in station_values]
                        )

                    st.markdown("**Station locations (manual or demo)**")
                    relocate_to_black_sea = False
                    if relocate_available:
                        relocate_to_black_sea = st.checkbox(
                            "Relocate to Black Sea (demo only)",
                            value=True
                        )
                    station_df = st.data_editor(
                        st.session_state[station_state_key],
                        column_config={
                            "lat": st.column_config.NumberColumn("lat", help="Latitude"),
                            "lon": st.column_config.NumberColumn("lon", help="Longitude")
                        },
                        use_container_width=True,
                        num_rows="dynamic"
                    )
                    st.session_state[station_state_key] = station_df

                    plot_df = station_df.copy()
                    plot_df["lat"] = pd.to_numeric(plot_df["lat"], errors="coerce")
                    plot_df["lon"] = pd.to_numeric(plot_df["lon"], errors="coerce")
                    if relocate_to_black_sea:
                        demo_coords = [(43.5, 33.8), (44.5, 36.2), (42.4, 28.8), (45.1, 31.1)]
                        for idx, row in plot_df.iterrows():
                            if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                                lat, lon = demo_coords[idx % len(demo_coords)]
                                plot_df.at[idx, "lat"] = lat
                                plot_df.at[idx, "lon"] = lon
                        st.caption("Markers are relocated for Black Sea map demo only.")

                    plot_df = plot_df.dropna(subset=["lat", "lon"])
                    if not plot_df.empty:
                        scope = "europe" if relocate_to_black_sea else "world"
                        fig = px.scatter_geo(
                            plot_df,
                            lat="lat",
                            lon="lon",
                            hover_name="name",
                            scope=scope
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Add lat/lon values to render the map.")

                    st.markdown("**Time-series preview**")
                    ts_station = st.selectbox("Station", station_values)
                    ts_field = st.selectbox("Field", sorted(df_noaa["_field"].dropna().unique().tolist()))
                    ts_df = df_noaa[(df_noaa[tag_key] == ts_station) & (df_noaa["_field"] == ts_field)]
                    if not ts_df.empty:
                        ts_fig = px.line(ts_df, x="_time", y="_value", title=f"{ts_station} — {ts_field}")
                        st.plotly_chart(ts_fig, use_container_width=True)
                    else:
                        st.info("No data for the selected station/field.")
    except Exception as exc:
        st.error(f"Influx fetch failed: {exc}")
