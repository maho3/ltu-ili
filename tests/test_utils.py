import camb
from ili.utils.cosmology import Cosmology


def test_cosmology():
    """Test cosmology class."""

    cosmo = Cosmology()
    c = cosmo.get_cosmology()
    assert isinstance(c, camb.results.CAMBdata)
    npoints = 200
    kh, z, pk = cosmo.get_matter_power_spectrum(npoints=npoints)
    assert kh.shape == (npoints,)
    assert pk.shape == (len(z), npoints)

    return
