package lib

import "golang.org/x/exp/slices"

func InSlice(ele string, sl []string) bool {
	return slices.Contains(sl, ele)
}

func IsSubSlice(subSl []string, sl []string) bool {
	for i, s := range sl {
		if subSl[0] == s {
			for j := 1; j < len(subSl); j++ {
				if subSl[j] != sl[i+j] {
					return false
				}
			}
			return true
		}
	}
	return false
}
