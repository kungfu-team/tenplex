package uri

import (
	"net/url"
	"path"
	"strings"
)

func AppendPath(a, b string) string {
	b = strings.TrimLeft(b, `/`)
	u, err := url.Parse(a)
	if err != nil {
		return path.Join(a, b)
	}
	u.Path = path.Join(u.Path, b)
	return u.String()
}
