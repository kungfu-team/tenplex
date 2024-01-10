package statetransform

import "testing"

func TestVocabSizePadding(t *testing.T) {
	mp := 2
	s := VocabSizePadding(30524, mp)
	t.Logf("vocab size with padding with MP %d: %d", mp, s)
	mp = 4
	s = VocabSizePadding(30524, mp)
	t.Logf("vocab size with padding with MP %d: %d", mp, s)
}
