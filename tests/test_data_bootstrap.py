import hashlib
import io
import tarfile
import zipfile

import pytest

from zcp_test.data.bootstrap import (
    BENCHMARK_ASSETS,
    BUILTIN_ASSETS,
    AssetState,
    BootstrapAsset,
    asset_checklist,
    bootstrap_data,
    resumable_http_download,
    safe_extract_tar,
    safe_extract_zip,
)


class FakeResponse(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {}

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_builtin_manifest_covers_supported_benchmarks():
    assert set(BENCHMARK_ASSETS) == {
        "nasbench101",
        "nasbench201",
        "nats_tss",
        "nats_sss",
        "transnasbench101",
        "nasbench301_surrogate",
        "vitbench101",
    }
    assert set(asset for assets in BENCHMARK_ASSETS.values() for asset in assets) == set(
        BUILTIN_ASSETS
    )
    assert all(asset.version and asset.path for asset in BUILTIN_ASSETS.values())


def test_checklist_distinguishes_missing_partial_invalid_and_ready(tmp_path):
    expected = hashlib.sha256(b"complete").hexdigest()
    assets = {
        "item": BootstrapAsset("item", "1", "item.bin", ("https://example/item",), expected)
    }
    assert asset_checklist(tmp_path, assets)[0].state is AssetState.MISSING

    (tmp_path / "item.bin.part").write_bytes(b"part")
    partial = asset_checklist(tmp_path, assets)[0]
    assert partial.state is AssetState.PARTIAL
    assert partial.partial_bytes == 4

    (tmp_path / "item.bin").write_bytes(b"wrong")
    assert asset_checklist(tmp_path, assets)[0].state is AssetState.INVALID
    (tmp_path / "item.bin").write_bytes(b"complete")
    assert asset_checklist(tmp_path, assets)[0].ready


def test_http_download_resumes_a_partial_file(tmp_path):
    destination = tmp_path / "asset.bin"
    (tmp_path / "asset.bin.part").write_bytes(b"abc")
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return FakeResponse(b"def", status=206, headers={"Content-Range": "bytes 3-5/6"})

    result = resumable_http_download(
        "https://example.test/asset",
        destination,
        sha256=hashlib.sha256(b"abcdef").hexdigest(),
        opener=opener,
    )

    assert result.read_bytes() == b"abcdef"
    assert requests[0][0].get_header("Range") == "bytes=3-"
    assert not (tmp_path / "asset.bin.part").exists()


def test_http_download_restarts_when_server_ignores_range(tmp_path):
    destination = tmp_path / "asset.bin"
    (tmp_path / "asset.bin.part").write_bytes(b"stale")

    def opener(request, *, timeout):
        return FakeResponse(b"fresh", status=200)

    resumable_http_download("https://example.test/asset", destination, opener=opener)
    assert destination.read_bytes() == b"fresh"


def test_http_download_quarantines_bad_checksum_partial(tmp_path):
    destination = tmp_path / "asset.bin"

    def opener(request, *, timeout):
        return FakeResponse(b"bad")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        resumable_http_download(
            "https://example.test/asset",
            destination,
            sha256=hashlib.sha256(b"good").hexdigest(),
            opener=opener,
        )
    assert not (tmp_path / "asset.bin.part").exists()
    assert len(list(tmp_path.glob("asset.bin.part.invalid-*"))) == 1
    assert not destination.exists()


def test_http_download_preserves_incomplete_response_for_resume(tmp_path):
    destination = tmp_path / "asset.bin"

    def opener(request, *, timeout):
        return FakeResponse(b"half", headers={"Content-Length": "8"})

    with pytest.raises(OSError, match="Incomplete HTTP response"):
        resumable_http_download("https://example.test/asset", destination, opener=opener)
    assert (tmp_path / "asset.bin.part").read_bytes() == b"half"
    assert not destination.exists()


def _write_tar(path, members):
    with tarfile.open(path, "w") as archive:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode()
                archive.addfile(info)


def test_safe_tar_extracts_regular_files(tmp_path):
    archive = tmp_path / "fixture.tar"
    _write_tar(archive, [("bundle/data.txt", b"payload", "file")])

    paths = safe_extract_tar(archive, tmp_path / "output")

    assert (tmp_path / "output/bundle/data.txt").read_bytes() == b"payload"
    assert tmp_path / "output/bundle/data.txt" in paths


def test_safe_zip_extracts_regular_files_and_rejects_traversal(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("bundle/data.txt", "payload")
    paths = safe_extract_zip(archive, tmp_path / "output")
    assert (tmp_path / "output/bundle/data.txt").read_text() == "payload"
    assert tmp_path / "output/bundle/data.txt" in paths

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as output:
        output.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="safe relative|Unsafe|path"):
        safe_extract_zip(unsafe, tmp_path / "unsafe-output")


@pytest.mark.parametrize(
    "name,kind",
    [("../escape.txt", "file"), ("/absolute.txt", "file"), ("link", "symlink")],
)
def test_safe_tar_rejects_traversal_and_links(tmp_path, name, kind):
    archive = tmp_path / "malicious.tar"
    _write_tar(archive, [(name, b"outside" if kind == "symlink" else b"bad", kind)])

    with pytest.raises(ValueError, match="Unsafe tar"):
        safe_extract_tar(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_bootstrap_orchestration_is_injectable_and_idempotent(tmp_path):
    asset = BootstrapAsset(
        "fixture",
        "1",
        "installed/payload.txt",
        ("https://mirror.invalid/fixture.tar",),
        archive="tar",
        archive_name="fixture.tar",
    )
    assets = {asset.asset_id: asset}
    calls = []

    def downloader(url, destination, *, sha256):
        calls.append((url, destination, sha256))
        _write_tar(destination, [("installed/payload.txt", b"ready", "file")])
        return destination

    first = bootstrap_data(tmp_path, assets=assets, downloader=downloader)
    second = bootstrap_data(tmp_path, assets=assets, downloader=downloader)

    assert first.ok and second.ok
    assert first.by_id()["fixture"].state is AssetState.READY
    assert (tmp_path / "installed/payload.txt").read_bytes() == b"ready"
    assert len(calls) == 1


def test_bootstrap_quarantines_corrupt_installed_file(tmp_path):
    expected = hashlib.sha256(b"valid").hexdigest()
    asset = BootstrapAsset(
        "fixture", "1", "fixture.bin", ("https://example.invalid/fixture",), expected
    )
    (tmp_path / "fixture.bin").write_bytes(b"corrupt")

    def downloader(url, destination, *, sha256):
        destination.write_bytes(b"valid")
        return destination

    result = bootstrap_data(tmp_path, assets={asset.asset_id: asset}, downloader=downloader)

    assert result.ok
    assert (tmp_path / "fixture.bin").read_bytes() == b"valid"
    assert len(list(tmp_path.glob("fixture.bin.invalid-*"))) == 1


def test_bootstrap_marks_manual_sources_unavailable(tmp_path):
    asset = BootstrapAsset("manual", "1", "manual/data.pth", source_page="https://source")
    result = bootstrap_data(tmp_path, assets={asset.asset_id: asset})

    assert not result.ok
    assert result.items[0].state is AssetState.UNAVAILABLE
    assert "https://source" in result.items[0].detail
