package scheduler

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"testing"
)

func TestScheduler(t *testing.T) {
	ip := "localhost"
	port := 22222
	url := fmt.Sprintf("http://%s:%d/stop", ip, port)

	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("error %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("POST failed, status code: %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("error %v", err)
	}
	t.Logf("body %s", string(body))
}

func TestNextLowerPowTwo(t *testing.T) {
	x := nextLowerPowTwo(33)
	if x == 32 {
		t.Logf("success")
		return
	}
	t.Logf("failed")
}

func TestPlayground(t *testing.T) {
	n := "iter"
	splitName := strings.SplitN(n, ".", 2)
	log.Printf("%d", len(splitName))
}
