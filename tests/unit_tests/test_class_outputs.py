"""Exercise the full extraction function without requiring a CLASS build."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from cosmosis.datablock import DataBlock, names

CLASS_SOURCE = Path(os.environ.get('CLASS_INTERFACE_SOURCE',
    str(Path(__file__).parents[2] / 'boltzmann/class/class_interface.py')))
spec = importlib.util.spec_from_file_location('class_outputs_under_test', CLASS_SOURCE)
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)

@pytest.mark.parametrize('save_linear', [False, True])
@pytest.mark.parametrize('save_cb', [False, True])
def test_nonlinear_grid_independent_of_optional_linear_outputs(save_linear, save_cb):
    block = DataBlock()
    h = 0.7
    block[names.cosmological_parameters, 'h0'] = h
    calls = []
    def pk(k,z):
        calls.append((k,z))
        return 100 + 2*k + 3*z
    c = SimpleNamespace(
        pars={'output':'mPk'}, nonlinear_method=1,
        pk=pk, pk_lin=lambda k,z: 10+k+z, pk_cb_lin=lambda k,z: 20+k+z,
        sigma8=lambda: .8, scale_independent_growth_factor=lambda z: 1/(1+z),
        scale_independent_growth_factor_f=lambda z: .5,
        effective_f_sigma8=lambda z: .4, sigma=lambda *a,**kw: .8,
        luminosity_distance=lambda z: 100*z*(1+z),
        angular_distance=lambda z: 100*z/(1+z),
        comoving_distance=lambda z: 100*z, Hubble=lambda z: .001,
        age=lambda: 13.8, rs_drag=lambda: 147.)
    config=dict(kmax=1.,nk=4,zmax=.02,background_only=False,cmb=False,
                save_matter_power_lin=save_linear,save_cdm_baryon_power_lin=save_cb)
    interface.get_class_outputs(block,c,config)
    z,kh,p=block.get_grid('matter_power_nl','z','k_h','p_k')
    assert p.shape==(3,4)
    assert len(calls)==12
    np.testing.assert_allclose(p,(100+2*kh[None,:]*h+3*z[:,None])*h**3,rtol=1e-14,atol=0)
    assert block.has_section('matter_power_lin')==save_linear
    assert block.has_section('cdm_baryon_power_lin')==save_cb
    if save_linear:
        _,_,linear=block.get_grid('matter_power_lin','z','k_h','p_k')
        np.testing.assert_allclose(linear,(10+kh[None,:]*h+z[:,None])*h**3,rtol=1e-14)
    if save_cb:
        _,_,cb=block.get_grid('cdm_baryon_power_lin','z','k_h','p_k')
        np.testing.assert_allclose(cb,(20+kh[None,:]*h+z[:,None])*h**3,rtol=1e-14)
