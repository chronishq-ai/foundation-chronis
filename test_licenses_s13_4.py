import json
from pathlib import Path
import pytest

from chronis_ml.ops import check_licenses, LicenseRequiresApprovalError


def test_boundary_copyleft_rejected(tmp_path):
    """Confirm that any copyleft license (GPL, AGPL, LGPL) for a boundary=true package raises ValueError."""
    # Write a requirements.txt with a mock package
    reqs = "fake-boundary==1.0.0\n"
    tmp_path.joinpath("requirements.txt").write_text(reqs)
    
    # GPL-3.0 on a boundary=true package
    licenses = [
        {"name": "fake-boundary", "version": "1.0.0", "license": "GPL-3.0", "boundary": True}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses))
    
    with pytest.raises(ValueError) as exc:
        check_licenses(tmp_path)
    assert "strictly forbidden inside trusted boundary" in str(exc.value)

    # LGPL-3.0 on a boundary=true package
    licenses = [
        {"name": "fake-boundary", "version": "1.0.0", "license": "LGPL-3.0", "boundary": True}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses))
    
    with pytest.raises(ValueError) as exc:
        check_licenses(tmp_path)
    assert "strictly forbidden inside trusted boundary" in str(exc.value)


def test_peripheral_copyleft_rejected(tmp_path):
    """Confirm that copyleft licenses on boundary=false (peripheral) packages are also rejected,
    because they are not present in OK_LICENSES.
    """
    reqs = "fake-peripheral==1.0.0\n"
    tmp_path.joinpath("requirements.txt").write_text(reqs)
    
    # GPL-3.0 on a boundary=false package
    licenses = [
        {"name": "fake-peripheral", "version": "1.0.0", "license": "GPL-3.0", "boundary": False}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses))
    
    with pytest.raises(ValueError) as exc:
        check_licenses(tmp_path)
    assert "is not in OK_LICENSES" in str(exc.value)


def test_boundary_lower_tolerance(tmp_path):
    """Confirm that boundary packages reject licenses like MPL-2.0, even though
    they are allowed for peripheral packages (illustrating lower tolerance).
    """
    # 1. Permitted on peripheral
    reqs = "fake-peripheral==1.0.0\n"
    tmp_path.joinpath("requirements.txt").write_text(reqs)
    licenses = [
        {"name": "fake-peripheral", "version": "1.0.0", "license": "MPL-2.0", "boundary": False}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses))
    
    # Should not raise any error
    check_licenses(tmp_path)

    # 2. Rejected on boundary
    reqs_b = "fake-boundary==1.0.0\n"
    tmp_path.joinpath("requirements.txt").write_text(reqs_b)
    licenses_b = [
        {"name": "fake-boundary", "version": "1.0.0", "license": "MPL-2.0", "boundary": True}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses_b))
    
    with pytest.raises(ValueError) as exc:
        check_licenses(tmp_path)
    assert "does not meet lower tolerance standard for trusted boundary" in str(exc.value)


def test_audeering_research_license_approval(tmp_path):
    """Confirm check_licenses() raises LicenseRequiresApprovalError on a boundary=true package
    with license="audEERING-Research-License" unless the approval is satisfied.
    """
    reqs = "fake-opensmile==2.5.0\n"
    tmp_path.joinpath("requirements.txt").write_text(reqs)
    licenses = [
        {"name": "fake-opensmile", "version": "2.5.0", "license": "audEERING-Research-License", "boundary": True}
    ]
    tmp_path.joinpath("licenses.json").write_text(json.dumps(licenses))

    # 1. Without approved_with_conditions.json, it should raise LicenseRequiresApprovalError
    with pytest.raises(LicenseRequiresApprovalError) as exc:
        check_licenses(tmp_path)
    assert "requires a paid commercial license from audEERING" in str(exc.value)

    # 2. With approved_with_conditions.json containing approval, it should pass
    conds = [
        {
            "name": "fake-opensmile",
            "approved": True,
            "commercial_license_reference": "purchased commercial license reference #12345"
        }
    ]
    tmp_path.joinpath("approved_with_conditions.json").write_text(json.dumps(conds))
    
    # Should pass without raising
    check_licenses(tmp_path)
