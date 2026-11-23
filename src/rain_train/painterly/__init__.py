"""Anti-code-look Studio Ghibli / Makoto Shinkai painterly rendering engine."""

from .clouds import render_cumulus_cathedral, render_sky_gradient
from .ink import (
    apply_line_boil,
    draw_lantern,
    draw_petals_and_motes,
    draw_station_perspective,
    draw_station_silhouette,
    draw_train_perspective,
    draw_train_profile,
    draw_umbrella_figure,
)
from .optics import (
    apply_atmospheric_haze,
    apply_filmic_grade,
    apply_halation_bloom,
    apply_vignette,
    render_aperture_bokeh,
    render_god_rays,
)
from .paper_grain import (
    apply_gouache_brushwork,
    apply_paper_grain,
    apply_pigment_granulation,
    generate_paper_grain,
)
from .weather import (
    apply_puddle_reflection,
    apply_sea_reflection,
    render_macro_window_droplets,
    render_sea_wake,
    render_slanted_rain,
    render_steam_plume,
)

__all__ = [
    "apply_atmospheric_haze",
    "apply_filmic_grade",
    "apply_gouache_brushwork",
    "apply_halation_bloom",
    "apply_line_boil",
    "apply_paper_grain",
    "apply_pigment_granulation",
    "apply_puddle_reflection",
    "apply_sea_reflection",
    "apply_vignette",
    "draw_lantern",
    "draw_petals_and_motes",
    "draw_station_perspective",
    "draw_station_silhouette",
    "draw_train_perspective",
    "draw_train_profile",
    "draw_umbrella_figure",
    "generate_paper_grain",
    "render_aperture_bokeh",
    "render_cumulus_cathedral",
    "render_god_rays",
    "render_macro_window_droplets",
    "render_sea_wake",
    "render_sky_gradient",
    "render_slanted_rain",
    "render_steam_plume",
]
