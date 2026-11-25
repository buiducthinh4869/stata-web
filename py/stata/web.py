"""Synchronous HTTP fetch inside a Pyodide web worker (used by `use "https://..."`)."""


def fetch_bytes(url):
    try:
        import js  # Pyodide only
    except ImportError as e:
        raise RuntimeError('network access is only available in the browser') from e
    xhr = js.XMLHttpRequest.new()
    xhr.open('GET', url, False)
    xhr.responseType = 'arraybuffer'
    xhr.send(None)
    if xhr.status != 200:
        raise RuntimeError(f'HTTP {xhr.status}')
    return bytes(js.Uint8Array.new(xhr.response).to_py())
