package uri

import (
	"log"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

func (o *Opener) loadAllSAS(prefix string) {
	fs, err := filepath.Glob(path.Join(prefix, `*.sas`))
	if err != nil {
		log.Printf("loadAllSAS: %v", err)
		return
	}
	for _, f := range fs {
		o.loadSAS(f)
	}
}

func (o *Opener) loadSAS(filename string) {
	bs, err := os.ReadFile(filename)
	if err != nil {
		return
	}
	sa := strings.TrimSuffix(path.Base(filename), `.sas`)
	sas := strings.TrimPrefix(strings.TrimSpace(string(bs)), `?`)
	if err := checkSAS(filename, sas); err != nil {
		log.Printf("invalid SAS %s: %v", filename, err)
		return
	}
	o.SetSAS(sa, sas)
	// log.Printf("loaded SAS for %s %s", sa, sas)
}

func (o *Opener) SetSAS(sa, sas string) {
	o.azSAS[sa] = sas
}

func (o *Opener) getSAS(sa, container string) string {
	if sas, ok := o.azSAS[sa]; ok {
		return sas
	}
	return ""
}

func (o *Opener) addAzureCreds(u url.URL) url.URL {
	// Host: <storage-account>.blob.core.windows.net
	hostParts := strings.SplitN(u.Host, `.`, 2)
	if len(hostParts) != 2 || hostParts[1] != azBlobDomain {
		return u
	}
	pathParts := removeEmpty(strings.Split(u.Path, `/`))
	if len(pathParts) <= 0 {
		return u
	}
	if len(u.RawQuery) == 0 {
		u.RawQuery += "&"
	}
	u.RawQuery += o.getSAS(hostParts[0], pathParts[0])
	return u
}

func removeEmpty(a []string) []string {
	var b []string
	for _, s := range a {
		if len(s) > 0 {
			b = append(b, s)
		}
	}
	return b
}
