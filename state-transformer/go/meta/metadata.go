package meta

type Metadata struct {
	SourceRankMap    *RankMap
	TargetRankMap    *RankMap
	SourceStructs    map[int]map[string]interface{}
	TargetStructs    map[int]map[string]interface{}
	SourceGroupSizes map[int]map[int]int
	TargetGroupSizes map[int]map[int]int
	SourceModelKeys  map[int][][]string
	TargetModelKeys  map[int][][]string
}
