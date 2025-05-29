module @Sora {
  func.func @main(%arg0: tensor<2x11340x1152xf16>, %arg1: tensor<1x224x1152xf16>, %arg2: tensor<2x6912xf16>) -> tensor<2x11340x1152xf16> {
    %0 = "sora.None"() : () -> none
    %1 = "sora.View"(%arg1) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %2 = "sora.Convert"(%1) <{dynamic_scale = true}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xsi8>
    %3 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %4 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %5 = "sora.Elementwise"(%3, %4) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %6:6 = "sora.Split"(%5) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %7 = "sora.View"(%arg0) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %8 = "sora.Layernorm"(%7) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %9 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %10 = "sora.Elementwise"(%6#1, %9) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %11 = "sora.Elementwise"(%8, %10) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %12 = "sora.Elementwise"(%11, %6#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %13 = "sora.View"(%12) <{shape = [56, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<56x405x1152xsi8>
    %14 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %15 = "sora.Weight"() : () -> tensor<1152xf16>
    %16 = "sora.View"(%13) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %17 = "sora.LinearW8"(%16, %14, %15) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x1152xsi8>, tensor<1152xf16>) -> tensor<56x405x1152xf16>
    %18 = "sora.Weight"() : () -> tensor<1152xf16>
    %19 = "sora.View"(%13) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %20 = "sora.LinearW8"(%19, %arg0, %18) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x11340x1152xf16>, tensor<1152xf16>) -> tensor<56x405x1152xf16>
    %21 = "sora.Weight"() : () -> tensor<1152xf16>
    %22 = "sora.View"(%13) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %23 = "sora.LinearW8"(%22, %arg2, %21) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x6912xf16>, tensor<1152xf16>) -> tensor<56x405x1152xf16>
    %24 = "sora.View"(%17) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xf16>) -> tensor<56x405x16x72xf16>
    %25 = "sora.View"(%20) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xf16>) -> tensor<56x405x16x72xf16>
    %26 = "sora.View"(%23) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xf16>) -> tensor<56x405x16x72xf16>
    %27 = "sora.Transpose"(%24) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %28 = "sora.Transpose"(%25) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %29 = "sora.Transpose"(%26) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %30 = "sora.View"(%27) <{shape = [362880, 72]}> : (tensor<56x16x405x72xf16>) -> tensor<362880x72xf16>
    %31 = "sora.Rmsnorm"(%30) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<56x16x405x72xf16>
    %32 = "sora.View"(%28) <{shape = [362880, 72]}> : (tensor<56x16x405x72xf16>) -> tensor<362880x72xf16>
    %33 = "sora.Rmsnorm"(%32) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<56x16x405x72xsi8>
    %34 = "sora.Weight"() : () -> tensor<56x16x405x72xf16>
    %35 = "sora.Elementwise"(%31, %34) <{dynamic_scale = true, op_type = "div"}> : (tensor<56x16x405x72xf16>, tensor<56x16x405x72xf16>) -> tensor<56x16x405x72xsi8>
    %36 = "sora.MatmulW8"(%35, %33) : (tensor<56x16x405x72xsi8>, tensor<56x16x405x72xsi8>) -> tensor<56x16x405x405xf16>
    %37 = "sora.View"(%36) <{shape = [362880, 405]}> : (tensor<56x16x405x405xf16>) -> tensor<362880x405xf16>
    %38 = "sora.Softmax"(%37) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x405xf16>) -> tensor<56x16x405x405xsi8>
    %39 = "sora.Transpose"(%29) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<56x16x72x405xf16>
    %40 = "sora.View"(%39) <{shape = [64512, 405]}> : (tensor<56x16x72x405xf16>) -> tensor<64512x405xf16>
    %41 = "sora.Convert"(%40) <{dynamic_scale = true}> : (tensor<64512x405xf16>) -> tensor<56x16x72x405xsi8>
    %42 = "sora.MatmulW8"(%38, %41) : (tensor<56x16x405x405xsi8>, tensor<56x16x72x405xsi8>) -> tensor<56x16x405x72xf16>
    %43 = "sora.Transpose"(%42) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<56x405x16x72xf16>
    %44 = "sora.View"(%43) <{shape = [56, 405, 1152]}> : (tensor<56x405x16x72xf16>) -> tensor<56x405x1152xf16>
    %45 = "sora.View"(%44) <{shape = [22680, 1152]}> : (tensor<56x405x1152xf16>) -> tensor<22680x1152xf16>
    %46 = "sora.Convert"(%45) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<56x405x1152xsi8>
    %47 = "sora.View"(%46) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %48 = "sora.LinearW8"(%47, %6#0, %6#2) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<56x405x1152xf16>
    %49 = "sora.View"(%48) <{shape = [2, 11340, 1152]}> : (tensor<56x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %50 = "sora.Elementwise"(%49, %6#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %51 = "sora.Elementwise"(%arg0, %50) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %52 = "sora.View"(%51) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %53 = "sora.Convert"(%52) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xsi8>
    %54 = "sora.View"(%53) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %55 = "sora.LinearW8"(%54, %6#3, %6#5) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %56 = "sora.View"(%2) <{shape = [224, 1152]}> : (tensor<1x224x1152xsi8>) -> tensor<224x1152xsi8>
    %57 = "sora.LinearW8"(%56, %8, %11) <{do_bias = true}> : (tensor<224x1152xsi8>, tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<1x224x1152xf16>
    %58 = "sora.View"(%2) <{shape = [224, 1152]}> : (tensor<1x224x1152xsi8>) -> tensor<224x1152xsi8>
    %59 = "sora.LinearW8"(%58, %12, %13) <{do_bias = true}> : (tensor<224x1152xsi8>, tensor<2x11340x1152xsi8>, tensor<56x405x1152xsi8>) -> tensor<1x224x1152xf16>
    %60 = "sora.View"(%55) <{shape = [1, 22680, 16, 72]}> : (tensor<2x11340x1152xf16>) -> tensor<1x22680x16x72xf16>
    %61 = "sora.View"(%57) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %62 = "sora.View"(%59) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %63 = "sora.Transpose"(%60) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %64 = "sora.Transpose"(%61) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %65 = "sora.Transpose"(%62) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %66 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %67 = "sora.Elementwise"(%63, %66) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %68 = "sora.View"(%64) <{shape = [3584, 72]}> : (tensor<1x16x224x72xf16>) -> tensor<3584x72xf16>
    %69 = "sora.Convert"(%68) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<1x16x224x72xsi8>
    %70 = "sora.MatmulW8"(%67, %69) : (tensor<1x16x22680x72xsi8>, tensor<1x16x224x72xsi8>) -> tensor<1x16x22680x224xf16>
    %71 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %72 = "sora.Elementwise"(%70, %71) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<1x16x22680x224xf16>
    %73 = "sora.View"(%72) <{shape = [362880, 224]}> : (tensor<1x16x22680x224xf16>) -> tensor<362880x224xf16>
    %74 = "sora.Softmax"(%73) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<1x16x22680x224xsi8>
    %75 = "sora.Transpose"(%65) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1x16x72x224xf16>
    %76 = "sora.View"(%75) <{shape = [1152, 224]}> : (tensor<1x16x72x224xf16>) -> tensor<1152x224xf16>
    %77 = "sora.Convert"(%76) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1x16x72x224xsi8>
    %78 = "sora.MatmulW8"(%74, %77) : (tensor<1x16x22680x224xsi8>, tensor<1x16x72x224xsi8>) -> tensor<1x16x22680x72xf16>
    %79 = "sora.Transpose"(%78) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %80 = "sora.View"(%79) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<2x11340x1152xf16>
    %81 = "sora.View"(%80) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %82 = "sora.Convert"(%81) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xsi8>
    %83 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %84 = "sora.View"(%82) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %85 = "sora.LinearW8"(%84, %83, %20) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x1152xsi8>, tensor<56x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %86 = "sora.Elementwise"(%51, %85) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %87 = "sora.View"(%86) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %88 = "sora.Layernorm"(%87) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %89 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %90 = "sora.Elementwise"(%6#4, %89) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %91 = "sora.Elementwise"(%88, %90) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %92 = "sora.Elementwise"(%91, %6#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %93 = "sora.View"(%92) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %94 = "sora.LinearW8"(%93, %24, %26) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x405x16x72xf16>, tensor<56x405x16x72xf16>) -> tensor<2x11340x4608xf16>
    %95 = "sora.View"(%94) <{shape = [22680, 4608]}> : (tensor<2x11340x4608xf16>) -> tensor<22680x4608xf16>
    %96 = "sora.Gelu"(%95) <{dynamic_scale = true}> : (tensor<22680x4608xf16>) -> tensor<2x11340x4608xsi8>
    %97 = "sora.View"(%96) <{shape = [22680, 4608]}> : (tensor<2x11340x4608xsi8>) -> tensor<22680x4608xsi8>
    %98 = "sora.LinearW8"(%97, %27, %29) <{do_bias = true}> : (tensor<22680x4608xsi8>, tensor<56x16x405x72xf16>, tensor<56x16x405x72xf16>) -> tensor<2x11340x1152xf16>
    %99 = "sora.Elementwise"(%98, %6#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %100 = "sora.Elementwise"(%86, %99) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %101 = "sora.Elementwise"(%85, %99) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %102 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %103 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %104 = "sora.Elementwise"(%102, %103) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %105:6 = "sora.Split"(%104) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %106 = "sora.View"(%100) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %107 = "sora.Layernorm"(%106) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %108 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %109 = "sora.Elementwise"(%105#1, %108) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %110 = "sora.Elementwise"(%107, %109) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %111 = "sora.Elementwise"(%110, %105#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %112 = "sora.View"(%111) <{shape = [2, 28, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<2x28x405x1152xsi8>
    %113 = "sora.Transpose"(%112) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x28x405x1152xsi8>) -> tensor<2x405x28x1152xsi8>
    %114 = "sora.View"(%113) <{shape = [810, 28, 1152]}> : (tensor<2x405x28x1152xsi8>) -> tensor<810x28x1152xsi8>
    %115 = "sora.Weight"() : () -> tensor<1152xf16>
    %116 = "sora.View"(%114) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %117 = "sora.LinearW8"(%116, %33, %115) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xsi8>, tensor<1152xf16>) -> tensor<810x28x1152xf16>
    %118 = "sora.View"(%114) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %119 = "sora.LinearW8"(%118, %35, %36) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xsi8>, tensor<56x16x405x405xf16>) -> tensor<810x28x1152xf16>
    %120 = "sora.View"(%114) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %121 = "sora.LinearW8"(%120, %38, %39) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x405xsi8>, tensor<56x16x72x405xf16>) -> tensor<810x28x1152xf16>
    %122 = "sora.View"(%117) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xf16>) -> tensor<810x28x16x72xf16>
    %123 = "sora.View"(%119) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xf16>) -> tensor<810x28x16x72xf16>
    %124 = "sora.View"(%121) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xf16>) -> tensor<810x28x16x72xf16>
    %125 = "sora.Transpose"(%122) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %126 = "sora.Transpose"(%123) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %127 = "sora.Transpose"(%124) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %128 = "sora.View"(%125) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %129 = "sora.Rmsnorm"(%128) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xf16>
    %130 = "sora.View"(%126) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %131 = "sora.Rmsnorm"(%130) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xf16>
    %132 = "sora.Rope"(%129, %42) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<810x16x28x72xf16>, tensor<56x16x405x72xf16>) -> tensor<810x16x28x72xf16>
    %133 = "sora.Rope"(%131, %42) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<810x16x28x72xf16>, tensor<56x16x405x72xf16>) -> tensor<810x16x28x72xf16>
    %134 = "sora.View"(%133) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %135 = "sora.Convert"(%134) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xsi8>
    %136 = "sora.Weight"() : () -> tensor<810x16x28x72xf16>
    %137 = "sora.Elementwise"(%132, %136) <{dynamic_scale = true, op_type = "div"}> : (tensor<810x16x28x72xf16>, tensor<810x16x28x72xf16>) -> tensor<810x16x28x72xsi8>
    %138 = "sora.MatmulW8"(%137, %135) : (tensor<810x16x28x72xsi8>, tensor<810x16x28x72xsi8>) -> tensor<810x16x28x28xf16>
    %139 = "sora.View"(%138) <{shape = [362880, 28]}> : (tensor<810x16x28x28xf16>) -> tensor<362880x28xf16>
    %140 = "sora.Softmax"(%139) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x28xf16>) -> tensor<810x16x28x28xsi8>
    %141 = "sora.Transpose"(%127) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<810x16x72x28xf16>
    %142 = "sora.View"(%141) <{shape = [933120, 28]}> : (tensor<810x16x72x28xf16>) -> tensor<933120x28xf16>
    %143 = "sora.Convert"(%142) <{dynamic_scale = true}> : (tensor<933120x28xf16>) -> tensor<810x16x72x28xsi8>
    %144 = "sora.MatmulW8"(%140, %143) : (tensor<810x16x28x28xsi8>, tensor<810x16x72x28xsi8>) -> tensor<810x16x28x72xf16>
    %145 = "sora.Transpose"(%144) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<810x28x16x72xf16>
    %146 = "sora.View"(%145) <{shape = [810, 28, 1152]}> : (tensor<810x28x16x72xf16>) -> tensor<810x28x1152xf16>
    %147 = "sora.View"(%146) <{shape = [22680, 1152]}> : (tensor<810x28x1152xf16>) -> tensor<22680x1152xf16>
    %148 = "sora.Convert"(%147) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<810x28x1152xsi8>
    %149 = "sora.View"(%148) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %150 = "sora.LinearW8"(%149, %43, %46) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x405x16x72xf16>, tensor<56x405x1152xsi8>) -> tensor<810x28x1152xf16>
    %151 = "sora.View"(%150) <{shape = [2, 405, 28, 1152]}> : (tensor<810x28x1152xf16>) -> tensor<2x405x28x1152xf16>
    %152 = "sora.Transpose"(%151) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x405x28x1152xf16>) -> tensor<2x28x405x1152xf16>
    %153 = "sora.View"(%152) <{shape = [2, 11340, 1152]}> : (tensor<2x28x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %154 = "sora.Elementwise"(%153, %105#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %155 = "sora.Elementwise"(%100, %154) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %156 = "sora.View"(%155) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %157 = "sora.Convert"(%156) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xsi8>
    %158 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %159 = "sora.View"(%157) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %160 = "sora.LinearW8"(%159, %158, %49) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x1152xsi8>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %161 = "sora.View"(%2) <{shape = [224, 1152]}> : (tensor<1x224x1152xsi8>) -> tensor<224x1152xsi8>
    %162 = "sora.LinearW8"(%161, %50, %53) <{do_bias = true}> : (tensor<224x1152xsi8>, tensor<2x11340x1152xf16>, tensor<2x11340x1152xsi8>) -> tensor<1x224x1152xf16>
    %163 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %164 = "sora.View"(%2) <{shape = [224, 1152]}> : (tensor<1x224x1152xsi8>) -> tensor<224x1152xsi8>
    %165 = "sora.LinearW8"(%164, %163, %57) <{do_bias = true}> : (tensor<224x1152xsi8>, tensor<1152x1152xsi8>, tensor<1x224x1152xf16>) -> tensor<1x224x1152xf16>
    %166 = "sora.View"(%160) <{shape = [1, 22680, 16, 72]}> : (tensor<2x11340x1152xf16>) -> tensor<1x22680x16x72xf16>
    %167 = "sora.View"(%162) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %168 = "sora.View"(%165) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %169 = "sora.Transpose"(%166) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %170 = "sora.Transpose"(%167) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %171 = "sora.Transpose"(%168) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %172 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %173 = "sora.Elementwise"(%169, %172) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %174 = "sora.View"(%170) <{shape = [3584, 72]}> : (tensor<1x16x224x72xf16>) -> tensor<3584x72xf16>
    %175 = "sora.Convert"(%174) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<1x16x224x72xsi8>
    %176 = "sora.MatmulW8"(%173, %175) : (tensor<1x16x22680x72xsi8>, tensor<1x16x224x72xsi8>) -> tensor<1x16x22680x224xf16>
    %177 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %178 = "sora.Elementwise"(%176, %177) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<1x16x22680x224xf16>
    %179 = "sora.View"(%178) <{shape = [362880, 224]}> : (tensor<1x16x22680x224xf16>) -> tensor<362880x224xf16>
    %180 = "sora.Softmax"(%179) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<1x16x22680x224xsi8>
    %181 = "sora.Transpose"(%171) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1x16x72x224xf16>
    %182 = "sora.View"(%181) <{shape = [1152, 224]}> : (tensor<1x16x72x224xf16>) -> tensor<1152x224xf16>
    %183 = "sora.Convert"(%182) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1x16x72x224xsi8>
    %184 = "sora.MatmulW8"(%180, %183) : (tensor<1x16x22680x224xsi8>, tensor<1x16x72x224xsi8>) -> tensor<1x16x22680x72xf16>
    %185 = "sora.Transpose"(%184) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %186 = "sora.View"(%185) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<2x11340x1152xf16>
    %187 = "sora.View"(%186) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %188 = "sora.Convert"(%187) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xsi8>
    %189 = "sora.View"(%188) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %190 = "sora.LinearW8"(%189, %59, %61) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1x224x1152xf16>, tensor<1x224x16x72xf16>) -> tensor<2x11340x1152xf16>
    %191 = "sora.Elementwise"(%155, %190) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %192 = "sora.View"(%191) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %193 = "sora.Layernorm"(%192) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %194 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %195 = "sora.Elementwise"(%105#4, %194) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %196 = "sora.Elementwise"(%193, %195) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %197 = "sora.Elementwise"(%196, %105#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %198 = "sora.View"(%197) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %199 = "sora.LinearW8"(%198, %63, %65) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1x16x22680x72xf16>, tensor<1x16x224x72xf16>) -> tensor<2x11340x4608xf16>
    %200 = "sora.View"(%199) <{shape = [22680, 4608]}> : (tensor<2x11340x4608xf16>) -> tensor<22680x4608xf16>
    %201 = "sora.Gelu"(%200) <{dynamic_scale = true}> : (tensor<22680x4608xf16>) -> tensor<2x11340x4608xsi8>
    %202 = "sora.Weight"() : () -> tensor<1152x4608xsi8>
    %203 = "sora.Weight"() : () -> tensor<1152xf16>
    %204 = "sora.View"(%201) <{shape = [22680, 4608]}> : (tensor<2x11340x4608xsi8>) -> tensor<22680x4608xsi8>
    %205 = "sora.LinearW8"(%204, %202, %203) <{do_bias = true}> : (tensor<22680x4608xsi8>, tensor<1152x4608xsi8>, tensor<1152xf16>) -> tensor<2x11340x1152xf16>
    %206 = "sora.Elementwise"(%205, %105#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %207 = "sora.Elementwise"(%191, %206) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %208 = "sora.Elementwise"(%190, %206) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    return %208 : tensor<2x11340x1152xf16>
  }
}

