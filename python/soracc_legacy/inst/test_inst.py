from .sora_inst import *
from bitstring import BitArray
import pytest

def test_ld_serilize_unserilize():
    ld = LDInst()
    ld.loop_1d = 61
    ld.wait = [PUType.MM]
    ret = ld.to_json()

    print(ret)

    ld2 = LDInst()
    ld2.from_json(ret)

    print(ld2.wait)
    assert ld2.loop_1d == 61
    assert PUType.MM in ld2.wait


def test_collector_serilize_unserilize():

    collect = InstCollector()
    ld = LDInst()
    ld.loop_1d = 31
    collect.add(ld)

    js_d = collect.to_json()
    new_collect = InstCollector()
    new_collect.from_json(js_d)

    assert len(new_collect.get_insts()) == 1, 'should have a ld'
    new_ld = new_collect.get_insts()[0]

    assert new_ld.loop_1d == 31, 'confirm inst field correct'


def test_ld_inst_binary():
    ld = LDInst()
    ld.wait += [PUType.ST, PUType.MISC, PUType.RS]
    ld.release += [PUType.ST, PUType.MISC, PUType.RS]
    ld.mode = LDMode.HBM2global
    ld.dst_bank_id = 0b10101

    ld_bins = ld.to_bin()
    # wait&release
    # assert ld_bins[18:28] == BitArray('0b1010110101')
    assert ld_bins[18:28] == BitArray('0b0101000001') 
    # dst_bank_id
    # assert ld_bins[18+32*4:23+32*4] == BitArray('0b10101')
    assert ld_bins[8+32*4:13+32*4] == BitArray('0b01010')
    # mode
    assert ld_bins[15:18] == BitArray('0b011')

def test_inst_binary():
    mm = MMInst()
    mm.wait += [PUType.LD]
    mm.release += [PUType.ST]

    rs = RSInst()
    rs.wait += [PUType.MM]
    rs.release += [PUType.SYS]

    ret = insts_to_bin((mm, rs))

def test_inst_collect():
    collect = InstCollector()
    ld = LDInst()
    ld.loop_1d = 31
    collect.add(ld)

    assert len(collect) == 1

    ld = collect[-1]
    assert ld.loop_1d == 31

    ld.length_1d = 1024

    collect[0] = ld

    assert collect[0].length_1d == 1024

