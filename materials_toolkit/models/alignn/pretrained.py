import requests
import os
import zipfile
from tqdm import tqdm
from .alignn import ALIGNN, ALIGNNConfig
import tempfile
import torch
from typing import Literal

models_name = Literal[
    "jv_formation_energy_peratom_alignn",
    "jv_optb88vdw_total_energy_alignn",
    "jv_optb88vdw_bandgap_alignn",
    "jv_mbj_bandgap_alignn",
    "jv_spillage_alignn",
    "jv_slme_alignn",
    "jv_bulk_modulus_kv_alignn",
    "jv_shear_modulus_gv_alignn",
    "jv_n-Seebeck_alignn",
    "jv_n-powerfact_alignn",
    "jv_magmom_oszicar_alignn",
    "jv_kpoint_length_unit_alignn",
    "jv_avg_elec_mass_alignn",
    "jv_avg_hole_mass_alignn",
    "jv_epsx_alignn",
    "jv_mepsx_alignn",
    "jv_max_efg_alignn",
    "jv_ehull_alignn",
    "jv_dfpt_piezo_max_dielectric_alignn",
    "jv_dfpt_piezo_max_dij_alignn",
    "jv_exfoliation_energy_alignn",
    "jv_supercon_tc_alignn",
    "jv_supercon_edos_alignn",
    "jv_supercon_debye_alignn",
    "jv_supercon_a2F_alignn",
    "mp_e_form_alignn",
    "mp_gappbe_alignn",
    "tinnet_O_alignn",
    "tinnet_N_alignn",
    "tinnet_OH_alignn",
    "AGRA_O_alignn",
    "AGRA_OH_alignn",
    "AGRA_CHO_alignn",
    "AGRA_CO_alignn",
    "AGRA_COOH_alignn",
    "qm9_U0_alignn",
    "qm9_U_alignn",
    "qm9_alpha_alignn",
    "qm9_gap_alignn",
    "qm9_G_alignn",
    "qm9_HOMO_alignn",
    "qm9_LUMO_alignn",
    "qm9_ZPVE_alignn",
    "hmof_co2_absp_alignn",
    "hmof_max_co2_adsp_alignn",
    "hmof_surface_area_m2g_alignn",
    "hmof_surface_area_m2cm3_alignn",
    "hmof_pld_alignn",
    "hmof_lcd_alignn",
    "hmof_void_fraction_alignn",
    "ocp2020_all",
    "ocp2020_100k",
    "ocp2020_10k",
    "jv_pdos_alignn",
]


def get_pretrained_alignn(
    model_name: models_name = "jv_formation_energy_peratom_alignn",
):
    """Get ALIGNN torch models from figshare."""
    # https://figshare.com/projects/ALIGNN_models/126478
    url = "https://figshare.com/ndownloader/files/31458679"
    output_features = 1
    config_params = {}
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    # print("Loading the zipfile...", zipfile.ZipFile(path).namelist())
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN(
        ALIGNNConfig(name="alignn", output_features=output_features, **config_params)
    )
    _, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(
        torch.load(filename, map_location="cpu")["model"], strict=False
    )
    model.eval()

    if os.path.exists(filename):
        os.remove(filename)

    return model
