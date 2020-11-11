# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: order.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='order.proto',
  package='order',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0border.proto\x12\x05order\" \n\x11\x41pplyOrderRequest\x12\x0b\n\x03sql\x18\x01 \x01(\t\"5\n\x12\x41pplyOrderResponse\x12\x0b\n\x03sql\x18\x01 \x01(\t\x12\x12\n\napplyOrder\x18\x02 \x01(\t2j\n\x19LogicalOptmizerApplyOrder\x12M\n\x14getApplyOrderRequest\x12\x18.order.ApplyOrderRequest\x1a\x19.order.ApplyOrderResponse\"\x00\x62\x06proto3'
)




_APPLYORDERREQUEST = _descriptor.Descriptor(
  name='ApplyOrderRequest',
  full_name='order.ApplyOrderRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='sql', full_name='order.ApplyOrderRequest.sql', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=22,
  serialized_end=54,
)


_APPLYORDERRESPONSE = _descriptor.Descriptor(
  name='ApplyOrderResponse',
  full_name='order.ApplyOrderResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='sql', full_name='order.ApplyOrderResponse.sql', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='applyOrder', full_name='order.ApplyOrderResponse.applyOrder', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=56,
  serialized_end=109,
)

DESCRIPTOR.message_types_by_name['ApplyOrderRequest'] = _APPLYORDERREQUEST
DESCRIPTOR.message_types_by_name['ApplyOrderResponse'] = _APPLYORDERRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ApplyOrderRequest = _reflection.GeneratedProtocolMessageType('ApplyOrderRequest', (_message.Message,), {
  'DESCRIPTOR' : _APPLYORDERREQUEST,
  '__module__' : 'order_pb2'
  # @@protoc_insertion_point(class_scope:order.ApplyOrderRequest)
  })
_sym_db.RegisterMessage(ApplyOrderRequest)

ApplyOrderResponse = _reflection.GeneratedProtocolMessageType('ApplyOrderResponse', (_message.Message,), {
  'DESCRIPTOR' : _APPLYORDERRESPONSE,
  '__module__' : 'order_pb2'
  # @@protoc_insertion_point(class_scope:order.ApplyOrderResponse)
  })
_sym_db.RegisterMessage(ApplyOrderResponse)



_LOGICALOPTMIZERAPPLYORDER = _descriptor.ServiceDescriptor(
  name='LogicalOptmizerApplyOrder',
  full_name='order.LogicalOptmizerApplyOrder',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=111,
  serialized_end=217,
  methods=[
  _descriptor.MethodDescriptor(
    name='getApplyOrderRequest',
    full_name='order.LogicalOptmizerApplyOrder.getApplyOrderRequest',
    index=0,
    containing_service=None,
    input_type=_APPLYORDERREQUEST,
    output_type=_APPLYORDERRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_LOGICALOPTMIZERAPPLYORDER)

DESCRIPTOR.services_by_name['LogicalOptmizerApplyOrder'] = _LOGICALOPTMIZERAPPLYORDER

# @@protoc_insertion_point(module_scope)