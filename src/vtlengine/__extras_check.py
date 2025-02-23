import importlib.util

EXTRAS_DOCS = "https://docs.vtlengine.meaningfuldata.eu/#installation"
ERROR_MESSAGE = (
    "The '{extra_name}' extra is required to run {extra_desc}. "
    "Please install it using 'pip install vtlengine[{extra_name}]' or "
    "install all extras with 'pip install vtlengine[all]'. "
    f"Check the documentation at: {EXTRAS_DOCS}"
)


def __check_s3_extra() -> None:
    package_loc = importlib.util.find_spec("s3fs")
    if package_loc is None:
        raise ImportError(
            ERROR_MESSAGE.format(extra_name="s3", extra_desc="over csv files using S3 URIs")
        ) from None
