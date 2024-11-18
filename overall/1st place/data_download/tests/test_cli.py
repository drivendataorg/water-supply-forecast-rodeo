import subprocess


def test_cli():
    result = subprocess.run(
        ["python", "-m", "wsfr_download", "bulk", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    assert result.returncode == 0
    assert "Usage: python -m wsfr_download bulk" in result.stdout
