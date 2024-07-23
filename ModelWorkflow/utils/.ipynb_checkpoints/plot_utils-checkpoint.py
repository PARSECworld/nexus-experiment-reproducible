import geopandas as gpd

def plot_square(axis, vertices : gpd.GeoSeries, color = "red", alpha = 0.8, linewidth = 1):
  """
    Takes a GeoSeries of four vertices and plots the corresponding square.
  """
  df_vertices = gpd.GeoDataFrame(geometry = vertices, crs = vertices.crs)
  for i in range(4):
    vertice_1 = df_vertices.loc[i].geometry
    vertice_2 = df_vertices.loc[(i + 1) % 4].geometry
    axis.plot([vertice_1.x, vertice_2.x], [vertice_1.y, vertice_2.y], alpha = alpha, linewidth = linewidth, color = color)


def plot_map(axis, df, color = "gray", alpha = 0.4, linewidth = 0.1):
  """
    Takes a GeoPandasDatagrame and plots its geometry map
  """
  df["geometry"].plot(ax = axis, facecolor = color, alpha = alpha, edgecolor = 'black', linewidth = linewidth)

  axis.tick_params(
      axis = 'both',
      which = 'both',
      bottom = False,
      top = False,
      labelbottom = False,
      right = False,
      left = False,
      labelleft = False
  )