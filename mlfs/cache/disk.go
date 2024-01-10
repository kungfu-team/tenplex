package cache

import (
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"log"
	"net/url"
	"os"
	"path"
	"strings"
	"sync"

	"github.com/kungfu-team/mlfs/hash"
	"github.com/kungfu-team/mlfs/uri"
)

var errInvalidMD5 = errors.New("invalid MD5")

type Cache struct {
	root     string
	fetching map[string]struct{} // URL -> local path
	cached   map[string]string   // URL -> local path
	mu       sync.RWMutex
}

func New(root string) *Cache {
	c := &Cache{
		root:     root,
		fetching: make(map[string]struct{}),
		cached:   make(map[string]string),
	}
	return c
}

func (c *Cache) IsFetching(url string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if _, ok := c.fetching[url]; ok {
		return true
	}
	return false
}

func (c *Cache) IsCached(url string) (string, bool) {
	local, ok := c.localPath(url)
	if !ok {
		return "", false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if _, ok := c.cached[url]; ok {
		return local, true
	}
	return "", false
}

func (c *Cache) setCached(url string, local string) {
	c.mu.Lock()
	c.cached[url] = local
	log.Printf("cached %s -> %s", url, local)
	c.mu.Unlock()
}

func (c *Cache) Fetch(url string, md5sum string) (string, error) {
	if c.IsFetching(url) {
		return "", fmt.Errorf("alreading fetching %s", url)
	}
	{
		c.mu.Lock()
		c.fetching[url] = struct{}{}
		c.mu.Unlock()
	}
	defer func() {
		c.mu.Lock()
		delete(c.fetching, url)
		c.mu.Unlock()
	}()
	local, ok := c.localPath(url)
	if !ok {
		return "", fmt.Errorf("can't cache %s", url)
	}
	if _, err := os.Stat(local); err == nil {
		if len(md5sum) > 0 {
			if s, err := hash.FileMD5(nil, local); err == nil && s == md5sum {
				log.Printf("local file %s has correct md5", local)
				c.setCached(url, local)
				return local, nil
			}
			log.Printf("local file %s has incorrect md5", local)
		}
	}
	if err := os.MkdirAll(path.Dir(local), os.ModePerm); err != nil {
		return "", err
	}
	lf, err := os.Create(local)
	if err != nil {
		return "", err
	}
	defer lf.Close()
	f, err := uri.Open(url)
	if err != nil {
		return "", err
	}
	h := md5.New()
	if _, err := io.Copy(lf, io.TeeReader(f, h)); err != nil {
		return "", err
	}
	if len(md5sum) > 0 {
		if s := fmt.Sprintf("%x", h.Sum(nil)); s != md5sum {
			return "", errInvalidMD5
		}
	}
	c.setCached(url, local)
	return local, nil
}

var stat Stat

func (c *Cache) OpenRange(url string, bgn, end int64) (io.ReadCloser, error) {
	if c != nil {
		if local, ok := c.IsCached(url); ok {
			stat.Hit()
			if LogCache {
				log.Printf("using local cache %s", local)
				stat.Log()
			}
			return uri.OpenRange(local, bgn, end)
		} else {
			stat.Miss()
			if LogCache {
				log.Printf("%s is not cached", url)
				stat.Log()
			}
		}
	}
	return uri.OpenRange(url, bgn, end)
}

func (c *Cache) localPath(s string) (string, bool) {
	u, err := url.Parse(s)
	if err != nil {
		return "", false
	}
	return path.Join(c.root, u.Host, strings.TrimPrefix(u.Path, `/`)), true
}
