package mapslice

import (
	"encoding/json"
	"fmt"
	"log"
	"testing"
)

func Test_1(t *testing.T) {
	ms := MapSlice{
		MapItem{"abc", 123, 0},
		MapItem{"def", 456, 0},
		MapItem{"ghi", 789, 0},
	}

	b, err := json.Marshal(ms)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(b))

	ms = MapSlice{}
	if err := json.Unmarshal(b, &ms); err != nil {
		log.Fatal(err)
	}

	fmt.Println(ms)
}
