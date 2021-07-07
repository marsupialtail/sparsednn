import struct
import numpy as np
EPS = 0.0000001
ST = 1

def dec_to_2s(number, width):
    if number > 0:
        result = bin(number)
        pad_l = width - (len(result) - 2)
        if pad_l < 0:
            return "Error not enough bits"
        else:
            return "0" * pad_l + result[2:]
    elif number == 0:
        return "0" * width
    else:
        return dec_to_2s(number + 2 ** width, width)

def float_to_hex(number):
    s = struct.pack('>f',number)
    bits = struct.unpack('>l',s)[0]
    return hex(int(dec_to_2s(bits,32),2)).upper().replace("X","f")

def half_to_hex(number):
    s = hex(np.float16(number).view('H'))[2:].zfill(4)
    return "0x" + s + s

def hex_to_bin(hex_string):
    scale = 16 ## equals to hexadecimal
    num_of_bits = 16
    return bin(int(hex_string, scale))[2:].zfill(num_of_bits)

def bin_to_half(bin_string):
    sign = int(bin_string[0])
    exponent = int(bin_string[1:6],2)
    mantissa = int(bin_string[6:],2)
    factor = 10 ** np.ceil(np.log10(mantissa))
    return (-1) ** sign * 2 ** (exponent - 15) * (1 + mantissa /factor)
