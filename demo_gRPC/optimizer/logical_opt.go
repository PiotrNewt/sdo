package main

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"

	"github.com/PiotrNewt/slow-down-the-optimizer/demo_gRPC/optimizer/order"
	"google.golang.org/grpc"
)

func decode(order string) []int {
	if len(order) == 0 {
		return []int{}
	}

	strs := strings.Split(order, "_")
	// if len(strs) != ruleNum { // error }
	res := make([]int, len(strs))
	for i := 0; i < len(strs); i++ {
		res[i], _ = strconv.Atoi(strs[i])
	}
	return res
}

func getApplyOrder(sql string) []int {
	// ip and part should can be configured
	connect, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
	if err != nil {
		log.Fatal(err)
	}
	defer connect.Close()

	client := order.NewLogicalOptmizerApplyOrderClient(connect)
	response, err := client.GetApplyOrderRequest(
		context.Background(),
		&order.ApplyOrderRequest{
			Sql: sql,
		})

	fmt.Println(response.GetSql())
	fmt.Println(response.GetApplyOrder())
	applyOrder := []int{}
	// applyOrder := decode(response.ApplyOrder)
	return applyOrder
}

func logicalRuleApply() {
	_ = getApplyOrder("select a,b from t1 join t2 on t1.a < 20 where t2.c like \"%ab123\";")
	// using order
}

func main() {
	logicalRuleApply()
}
