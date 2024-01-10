package uri

import (
	"errors"
	"net/http"
	"net/url"
	"os"
	"strconv"
)

type Info struct {
	Size int64
}

func (o *Opener) Stat(uri string) (*Info, error) {
	u, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	return o.stat(*u)
}

func (o *Opener) stat(u url.URL) (*Info, error) {
	switch u.Scheme {
	case "http":
		return o.statHTTP(u)
	case "https":
		return o.statHTTP(o.addAzureCreds(u))
	case "file":
		return statFile(u.Path)
	case "":
		return statFile(u.Path)
	}
	return nil, errUnsupportedURLScheme
}

func (o *Opener) statHTTP(u url.URL) (*Info, error) {
	req, err := http.NewRequest(http.MethodHead, u.String(), nil)
	if err != nil {
		return nil, err
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New(resp.Status)
	}
	if cl := resp.Header.Get(`Content-Length`); len(cl) > 0 {
		n, err := strconv.ParseInt(cl, 10, 64)
		if err != nil {
			return nil, err
		}
		return &Info{Size: n}, nil
	}
	return &Info{Size: -1}, nil
}

func statFile(name string) (*Info, error) {
	info, err := os.Stat(name)
	if err != nil {
		return nil, err
	}
	return &Info{Size: info.Size()}, nil
}

func Stat(uri string) (*Info, error) {
	return opener.Stat(uri)
}
