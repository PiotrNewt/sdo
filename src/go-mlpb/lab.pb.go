// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: lab.proto

/*
	Package mlpb is a generated protocol buffer package.

	It is generated from these files:
		lab.proto

	It has these top-level messages:
		NextApplyIdxRequest
		NextApplyIdxResponse
*/
package mlpb

import proto "github.com/golang/protobuf/proto"
import fmt "fmt"
import math "math"

import (
	context "golang.org/x/net/context"
	grpc "google.golang.org/grpc"
)

import io "io"

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion2 // please upgrade the proto package

type NextApplyIdxRequest struct {
	SqlFingerPrint string `protobuf:"bytes,1,opt,name=sqlFingerPrint,proto3" json:"sqlFingerPrint,omitempty"`
	Sql            string `protobuf:"bytes,2,opt,name=sql,proto3" json:"sql,omitempty"`
	Reward         int64  `protobuf:"varint,3,opt,name=reward,proto3" json:"reward,omitempty"`
	IsFinished     bool   `protobuf:"varint,4,opt,name=isFinished,proto3" json:"isFinished,omitempty"`
	Plan           string `protobuf:"bytes,5,opt,name=plan,proto3" json:"plan,omitempty"`
	Flag           string `protobuf:"bytes,6,opt,name=flag,proto3" json:"flag,omitempty"`
}

func (m *NextApplyIdxRequest) Reset()                    { *m = NextApplyIdxRequest{} }
func (m *NextApplyIdxRequest) String() string            { return proto.CompactTextString(m) }
func (*NextApplyIdxRequest) ProtoMessage()               {}
func (*NextApplyIdxRequest) Descriptor() ([]byte, []int) { return fileDescriptorLab, []int{0} }

func (m *NextApplyIdxRequest) GetSqlFingerPrint() string {
	if m != nil {
		return m.SqlFingerPrint
	}
	return ""
}

func (m *NextApplyIdxRequest) GetSql() string {
	if m != nil {
		return m.Sql
	}
	return ""
}

func (m *NextApplyIdxRequest) GetReward() int64 {
	if m != nil {
		return m.Reward
	}
	return 0
}

func (m *NextApplyIdxRequest) GetIsFinished() bool {
	if m != nil {
		return m.IsFinished
	}
	return false
}

func (m *NextApplyIdxRequest) GetPlan() string {
	if m != nil {
		return m.Plan
	}
	return ""
}

func (m *NextApplyIdxRequest) GetFlag() string {
	if m != nil {
		return m.Flag
	}
	return ""
}

type NextApplyIdxResponse struct {
	SqlFingerPrint string `protobuf:"bytes,1,opt,name=sqlFingerPrint,proto3" json:"sqlFingerPrint,omitempty"`
	Sql            string `protobuf:"bytes,2,opt,name=sql,proto3" json:"sql,omitempty"`
	RuleIdx        int64  `protobuf:"varint,3,opt,name=ruleIdx,proto3" json:"ruleIdx,omitempty"`
}

func (m *NextApplyIdxResponse) Reset()                    { *m = NextApplyIdxResponse{} }
func (m *NextApplyIdxResponse) String() string            { return proto.CompactTextString(m) }
func (*NextApplyIdxResponse) ProtoMessage()               {}
func (*NextApplyIdxResponse) Descriptor() ([]byte, []int) { return fileDescriptorLab, []int{1} }

func (m *NextApplyIdxResponse) GetSqlFingerPrint() string {
	if m != nil {
		return m.SqlFingerPrint
	}
	return ""
}

func (m *NextApplyIdxResponse) GetSql() string {
	if m != nil {
		return m.Sql
	}
	return ""
}

func (m *NextApplyIdxResponse) GetRuleIdx() int64 {
	if m != nil {
		return m.RuleIdx
	}
	return 0
}

func init() {
	proto.RegisterType((*NextApplyIdxRequest)(nil), "mlpb.NextApplyIdxRequest")
	proto.RegisterType((*NextApplyIdxResponse)(nil), "mlpb.NextApplyIdxResponse")
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// Client API for AutoLogicalRulesApply service

type AutoLogicalRulesApplyClient interface {
	GetNextApplyIdxRequest(ctx context.Context, in *NextApplyIdxRequest, opts ...grpc.CallOption) (*NextApplyIdxResponse, error)
}

type autoLogicalRulesApplyClient struct {
	cc *grpc.ClientConn
}

func NewAutoLogicalRulesApplyClient(cc *grpc.ClientConn) AutoLogicalRulesApplyClient {
	return &autoLogicalRulesApplyClient{cc}
}

func (c *autoLogicalRulesApplyClient) GetNextApplyIdxRequest(ctx context.Context, in *NextApplyIdxRequest, opts ...grpc.CallOption) (*NextApplyIdxResponse, error) {
	out := new(NextApplyIdxResponse)
	err := grpc.Invoke(ctx, "/mlpb.AutoLogicalRulesApply/getNextApplyIdxRequest", in, out, c.cc, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// Server API for AutoLogicalRulesApply service

type AutoLogicalRulesApplyServer interface {
	GetNextApplyIdxRequest(context.Context, *NextApplyIdxRequest) (*NextApplyIdxResponse, error)
}

func RegisterAutoLogicalRulesApplyServer(s *grpc.Server, srv AutoLogicalRulesApplyServer) {
	s.RegisterService(&_AutoLogicalRulesApply_serviceDesc, srv)
}

func _AutoLogicalRulesApply_GetNextApplyIdxRequest_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NextApplyIdxRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(AutoLogicalRulesApplyServer).GetNextApplyIdxRequest(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/mlpb.AutoLogicalRulesApply/GetNextApplyIdxRequest",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(AutoLogicalRulesApplyServer).GetNextApplyIdxRequest(ctx, req.(*NextApplyIdxRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _AutoLogicalRulesApply_serviceDesc = grpc.ServiceDesc{
	ServiceName: "mlpb.AutoLogicalRulesApply",
	HandlerType: (*AutoLogicalRulesApplyServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "getNextApplyIdxRequest",
			Handler:    _AutoLogicalRulesApply_GetNextApplyIdxRequest_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "lab.proto",
}

func (m *NextApplyIdxRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NextApplyIdxRequest) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.SqlFingerPrint) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.SqlFingerPrint)))
		i += copy(dAtA[i:], m.SqlFingerPrint)
	}
	if len(m.Sql) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.Sql)))
		i += copy(dAtA[i:], m.Sql)
	}
	if m.Reward != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintLab(dAtA, i, uint64(m.Reward))
	}
	if m.IsFinished {
		dAtA[i] = 0x20
		i++
		if m.IsFinished {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i++
	}
	if len(m.Plan) > 0 {
		dAtA[i] = 0x2a
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.Plan)))
		i += copy(dAtA[i:], m.Plan)
	}
	if len(m.Flag) > 0 {
		dAtA[i] = 0x32
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.Flag)))
		i += copy(dAtA[i:], m.Flag)
	}
	return i, nil
}

func (m *NextApplyIdxResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalTo(dAtA)
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NextApplyIdxResponse) MarshalTo(dAtA []byte) (int, error) {
	var i int
	_ = i
	var l int
	_ = l
	if len(m.SqlFingerPrint) > 0 {
		dAtA[i] = 0xa
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.SqlFingerPrint)))
		i += copy(dAtA[i:], m.SqlFingerPrint)
	}
	if len(m.Sql) > 0 {
		dAtA[i] = 0x12
		i++
		i = encodeVarintLab(dAtA, i, uint64(len(m.Sql)))
		i += copy(dAtA[i:], m.Sql)
	}
	if m.RuleIdx != 0 {
		dAtA[i] = 0x18
		i++
		i = encodeVarintLab(dAtA, i, uint64(m.RuleIdx))
	}
	return i, nil
}

func encodeVarintLab(dAtA []byte, offset int, v uint64) int {
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return offset + 1
}
func (m *NextApplyIdxRequest) Size() (n int) {
	var l int
	_ = l
	l = len(m.SqlFingerPrint)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	l = len(m.Sql)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	if m.Reward != 0 {
		n += 1 + sovLab(uint64(m.Reward))
	}
	if m.IsFinished {
		n += 2
	}
	l = len(m.Plan)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	l = len(m.Flag)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	return n
}

func (m *NextApplyIdxResponse) Size() (n int) {
	var l int
	_ = l
	l = len(m.SqlFingerPrint)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	l = len(m.Sql)
	if l > 0 {
		n += 1 + l + sovLab(uint64(l))
	}
	if m.RuleIdx != 0 {
		n += 1 + sovLab(uint64(m.RuleIdx))
	}
	return n
}

func sovLab(x uint64) (n int) {
	for {
		n++
		x >>= 7
		if x == 0 {
			break
		}
	}
	return n
}
func sozLab(x uint64) (n int) {
	return sovLab(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (m *NextApplyIdxRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLab
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: NextApplyIdxRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NextApplyIdxRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SqlFingerPrint", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SqlFingerPrint = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Sql", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Sql = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Reward", wireType)
			}
			m.Reward = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Reward |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field IsFinished", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.IsFinished = bool(v != 0)
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Plan", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Plan = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Flag", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Flag = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLab(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthLab
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *NextApplyIdxResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLab
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: NextApplyIdxResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NextApplyIdxResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field SqlFingerPrint", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.SqlFingerPrint = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Sql", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= (uint64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthLab
			}
			postIndex := iNdEx + intStringLen
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Sql = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RuleIdx", wireType)
			}
			m.RuleIdx = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLab
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RuleIdx |= (int64(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipLab(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if skippy < 0 {
				return ErrInvalidLengthLab
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipLab(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLab
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowLab
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
			return iNdEx, nil
		case 1:
			iNdEx += 8
			return iNdEx, nil
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowLab
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			iNdEx += length
			if length < 0 {
				return 0, ErrInvalidLengthLab
			}
			return iNdEx, nil
		case 3:
			for {
				var innerWire uint64
				var start int = iNdEx
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return 0, ErrIntOverflowLab
					}
					if iNdEx >= l {
						return 0, io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					innerWire |= (uint64(b) & 0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				innerWireType := int(innerWire & 0x7)
				if innerWireType == 4 {
					break
				}
				next, err := skipLab(dAtA[start:])
				if err != nil {
					return 0, err
				}
				iNdEx = start + next
			}
			return iNdEx, nil
		case 4:
			return iNdEx, nil
		case 5:
			iNdEx += 4
			return iNdEx, nil
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
	}
	panic("unreachable")
}

var (
	ErrInvalidLengthLab = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLab   = fmt.Errorf("proto: integer overflow")
)

func init() { proto.RegisterFile("lab.proto", fileDescriptorLab) }

var fileDescriptorLab = []byte{
	// 273 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xe2, 0xcc, 0x49, 0x4c, 0xd2,
	0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x62, 0xc9, 0xcd, 0x29, 0x48, 0x52, 0x5a, 0xcf, 0xc8, 0x25,
	0xec, 0x97, 0x5a, 0x51, 0xe2, 0x58, 0x50, 0x90, 0x53, 0xe9, 0x99, 0x52, 0x11, 0x94, 0x5a, 0x58,
	0x9a, 0x5a, 0x5c, 0x22, 0xa4, 0xc6, 0xc5, 0x57, 0x5c, 0x98, 0xe3, 0x96, 0x99, 0x97, 0x9e, 0x5a,
	0x14, 0x50, 0x94, 0x99, 0x57, 0x22, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0x19, 0x84, 0x26, 0x2a, 0x24,
	0xc0, 0xc5, 0x5c, 0x5c, 0x98, 0x23, 0xc1, 0x04, 0x96, 0x04, 0x31, 0x85, 0xc4, 0xb8, 0xd8, 0x8a,
	0x52, 0xcb, 0x13, 0x8b, 0x52, 0x24, 0x98, 0x15, 0x18, 0x35, 0x98, 0x83, 0xa0, 0x3c, 0x21, 0x39,
	0x2e, 0xae, 0xcc, 0x62, 0xb7, 0xcc, 0xbc, 0xcc, 0xe2, 0x8c, 0xd4, 0x14, 0x09, 0x16, 0x05, 0x46,
	0x0d, 0x8e, 0x20, 0x24, 0x11, 0x21, 0x21, 0x2e, 0x96, 0x82, 0x9c, 0xc4, 0x3c, 0x09, 0x56, 0xb0,
	0x51, 0x60, 0x36, 0x48, 0x2c, 0x2d, 0x27, 0x31, 0x5d, 0x82, 0x0d, 0x22, 0x06, 0x62, 0x2b, 0x65,
	0x71, 0x89, 0xa0, 0x3a, 0xb8, 0xb8, 0x20, 0x3f, 0xaf, 0x38, 0x95, 0x02, 0x17, 0x4b, 0x70, 0xb1,
	0x17, 0x95, 0xe6, 0xa4, 0x7a, 0xa6, 0x54, 0x40, 0x9d, 0x0c, 0xe3, 0x1a, 0x65, 0x71, 0x89, 0x3a,
	0x96, 0x96, 0xe4, 0xfb, 0xe4, 0xa7, 0x67, 0x26, 0x27, 0xe6, 0x04, 0x95, 0xe6, 0xa4, 0x16, 0x83,
	0xed, 0x15, 0x0a, 0xe4, 0x12, 0x4b, 0x4f, 0x2d, 0xc1, 0x16, 0x70, 0x92, 0x7a, 0xa0, 0x70, 0xd5,
	0xc3, 0x22, 0x25, 0x25, 0x85, 0x4d, 0x0a, 0xe2, 0x7a, 0x25, 0x06, 0x27, 0x81, 0x13, 0x8f, 0xe4,
	0x18, 0x2f, 0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0x71, 0xc6, 0x63, 0x39, 0x86, 0x24, 0x36,
	0x70, 0x44, 0x19, 0x03, 0x02, 0x00, 0x00, 0xff, 0xff, 0x66, 0x6a, 0xa9, 0x80, 0xb5, 0x01, 0x00,
	0x00,
}