package uri

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"

	"github.com/kungfu-team/tenplex/mlfs/closer"
)

const azBlobDomain = `blob.core.windows.net`

type Opener struct {
	client http.Client

	azSAS map[string]string
}

var opener Opener

func init() {
	var home = os.Getenv(`HOME`)
	opener.azSAS = make(map[string]string)
	opener.loadAllSAS(`/etc/mlfs`)
	opener.loadAllSAS(`/etc/mlfs/azure`)
	opener.loadAllSAS(path.Join(home, `.az`))
}

func SetSAS(sa, sas string) {
	opener.SetSAS(sa, sas)
}

// Open extends os.Open to support URI, and adds azure credential automatically
func (o *Opener) Open(uri string) (io.ReadCloser, error) {
	return o.OpenRange(uri, 0, -1)
}

func (o *Opener) OpenRange(uri string, bgn int64, end int64) (io.ReadCloser, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	return o.openRange(*u, bgn, end)
}

func (o *Opener) openRange(u url.URL, bgn int64, end int64) (io.ReadCloser, error) {
	switch u.Scheme {
	case "http":
		f, err := o.openHTTPRange(u, bgn, end)
		if err == nil {
			f = withHTTPTrace(f, bgn, end)
		}
		return f, err
	case "https":
		f, err := o.openHTTPRange(o.addAzureCreds(u), bgn, end)
		if err == nil {
			f = withHTTPTrace(f, bgn, end)
		}
		return f, err
	case "file":
		return openFileRange(u.Path, bgn, end)
	case "":
		return openFileRange(u.Path, bgn, end)
	}
	return nil, errUnsupportedURLScheme
}

var errUnsupportedURLScheme = errors.New("unsupported URL scheme")

func (o *Opener) openHTTPRange(u url.URL, bgn int64, end int64) (io.ReadCloser, error) {
	// log.Printf("openHTTPAt(%s, %d)", &u, off)
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	reqRange := fmt.Sprintf("bytes=%d-%d", bgn, end-1)
	if bgn > 0 {
		req.Header.Set(`Range`, reqRange)
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	if bgn > 0 {
		if resp.StatusCode != http.StatusPartialContent {
			resp.Body.Close()
			return nil, errors.New(resp.Status)
		}
		// FIXME: check header
		// Content-Range: bytes 90773020-90863468/145498622
		// Content-Length: 90449
		// log.Printf("openedHTTP: %s | %s", resp.Header.Get(`Content-Range`), resp.Header.Get(`Content-Length`))
	} else {
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, errors.New(resp.Status)
		}
		if end > 0 {
			return closer.ReadClose(io.LimitReader(resp.Body, int64(end-bgn)), resp.Body.Close), nil
		}
	}
	return resp.Body, nil
}

// Open extends os.Open to support URI, and adds azure credential automatically
func Open(uri string) (io.ReadCloser, error) {
	return opener.Open(uri)
}

func OpenRange(uri string, bgn int64, end int64) (io.ReadCloser, error) {
	return opener.OpenRange(uri, bgn, end)
}

func openFileRange(name string, bgn int64, end int64) (io.ReadCloser, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	if bgn > 0 {
		if _, err := f.Seek(int64(bgn), io.SeekStart); err != nil {
			f.Close()
			return nil, err
		}
	}
	if end > 0 {
		return closer.ReadClose(io.LimitReader(f, end-bgn), f.Close), nil
	}
	return fileReadRate.Trace(f), nil
	// return f, nil
}
