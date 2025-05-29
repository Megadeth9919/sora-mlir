module @Sora {
  func.func @main(%arg0: tensor<22680x1152xf16>, %arg1: tensor<224x1152xf16>, %arg2: tensor<2x6912xf16>) -> tensor<2x11340x1152xf16> {
    %0 = "sora.None"() : () -> none
    %1 = "sora.Convert"(%arg1) <{dynamic_scale = true}> : (tensor<224x1152xf16>) -> tensor<224x1152xf16>
    %2 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %3 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %4 = "sora.Elementwise"(%2, %3) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %5:6 = "sora.Split"(%4) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %6 = "sora.Layernorm"(%arg0) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %7 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %8 = "sora.Elementwise"(%5#1, %7) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %9 = "sora.Elementwise"(%6, %8) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %10 = "sora.Elementwise"(%9, %5#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %11 = "sora.View"(%10) <{shape = [56, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %12 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %13 = "sora.Weight"() : () -> tensor<1152xf16>
    %14 = "sora.LinearW8"(%11, %12, %13) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x1152xsi8>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %15 = "sora.Weight"() : () -> tensor<1152xf16>
    %16 = "sora.LinearW8"(%11, %arg0, %15) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<22680x1152xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %17 = "sora.Weight"() : () -> tensor<1152xf16>
    %18 = "sora.LinearW8"(%11, %arg2, %17) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x6912xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %19 = "sora.View"(%14) <{shape = [56, 405, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x16x72xf16>
    %20 = "sora.View"(%16) <{shape = [56, 405, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x16x72xf16>
    %21 = "sora.View"(%18) <{shape = [56, 405, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x16x72xf16>
    %22 = "sora.Transpose"(%19) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<362880x72xf16>
    %23 = "sora.Transpose"(%20) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<362880x72xf16>
    %24 = "sora.Transpose"(%21) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %25 = "sora.Rmsnorm"(%22) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %26 = "sora.Rmsnorm"(%23) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %27 = "sora.Weight"() : () -> tensor<56x16x405x72xf16>
    %28 = "sora.Elementwise"(%25, %27) <{dynamic_scale = true, op_type = "div"}> : (tensor<362880x72xf16>, tensor<56x16x405x72xf16>) -> tensor<56x16x405x72xsi8>
    %29 = "sora.MatmulW8"(%28, %26) : (tensor<56x16x405x72xsi8>, tensor<362880x72xf16>) -> tensor<362880x405xf16>
    %30 = "sora.Softmax"(%29) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x405xf16>) -> tensor<362880x405xf16>
    %31 = "sora.Transpose"(%24) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<64512x405xf16>
    %32 = "sora.Convert"(%31) <{dynamic_scale = true}> : (tensor<64512x405xf16>) -> tensor<64512x405xf16>
    %33 = "sora.MatmulW8"(%30, %32) : (tensor<362880x405xf16>, tensor<64512x405xf16>) -> tensor<56x16x405x72xf16>
    %34 = "sora.Transpose"(%33) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<56x405x16x72xf16>
    %35 = "sora.View"(%34) <{shape = [56, 405, 1152]}> : (tensor<56x405x16x72xf16>) -> tensor<22680x1152xf16>
    %36 = "sora.Convert"(%35) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %37 = "sora.LinearW8"(%36, %5#0, %5#2) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xf16>
    %38 = "sora.View"(%37) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %39 = "sora.Elementwise"(%38, %5#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %40 = "sora.Elementwise"(%arg0, %39) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %41 = "sora.Convert"(%40) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %42 = "sora.LinearW8"(%41, %5#3, %5#5) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xf16>
    %43 = "sora.LinearW8"(%1, %6, %9) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<224x1152xf16>
    %44 = "sora.LinearW8"(%1, %10, %11) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<2x11340x1152xsi8>, tensor<22680x1152xsi8>) -> tensor<224x1152xf16>
    %45 = "sora.View"(%42) <{shape = [1, 22680, 16, 72]}> : (tensor<22680x1152xf16>) -> tensor<1x22680x16x72xf16>
    %46 = "sora.View"(%43) <{shape = [1, 224, 16, 72]}> : (tensor<224x1152xf16>) -> tensor<1x224x16x72xf16>
    %47 = "sora.View"(%44) <{shape = [1, 224, 16, 72]}> : (tensor<224x1152xf16>) -> tensor<1x224x16x72xf16>
    %48 = "sora.Transpose"(%45) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %49 = "sora.Transpose"(%46) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<3584x72xf16>
    %50 = "sora.Transpose"(%47) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %51 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %52 = "sora.Elementwise"(%48, %51) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %53 = "sora.Convert"(%49) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<3584x72xf16>
    %54 = "sora.MatmulW8"(%52, %53) : (tensor<1x16x22680x72xsi8>, tensor<3584x72xf16>) -> tensor<1x16x22680x224xf16>
    %55 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %56 = "sora.Elementwise"(%54, %55) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<362880x224xf16>
    %57 = "sora.Softmax"(%56) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<362880x224xf16>
    %58 = "sora.Transpose"(%50) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1152x224xf16>
    %59 = "sora.Convert"(%58) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1152x224xf16>
    %60 = "sora.MatmulW8"(%57, %59) : (tensor<362880x224xf16>, tensor<1152x224xf16>) -> tensor<1x16x22680x72xf16>
    %61 = "sora.Transpose"(%60) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %62 = "sora.View"(%61) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<22680x1152xf16>
    %63 = "sora.Convert"(%62) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %64 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %65 = "sora.LinearW8"(%63, %64, %16) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<1152x1152xsi8>, tensor<22680x1152xsi8>) -> tensor<22680x1152xf16>
    %66 = "sora.Elementwise"(%40, %65) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %67 = "sora.Layernorm"(%66) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %68 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %69 = "sora.Elementwise"(%5#4, %68) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %70 = "sora.Elementwise"(%67, %69) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %71 = "sora.Elementwise"(%70, %5#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xsi8>
    %72 = "sora.LinearW8"(%71, %19, %21) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x405x16x72xf16>, tensor<56x405x16x72xf16>) -> tensor<22680x1152xsi8>
    %73 = "sora.Gelu"(%72) <{dynamic_scale = true}> : (tensor<22680x1152xsi8>) -> tensor<22680x1152xsi8>
    %74 = "sora.LinearW8"(%73, %22, %24) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<362880x72xf16>, tensor<56x16x405x72xf16>) -> tensor<22680x1152xsi8>
    %75 = "sora.Elementwise"(%74, %5#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xsi8>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %76 = "sora.Elementwise"(%66, %75) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %77 = "sora.Elementwise"(%65, %75) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %78 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %79 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %80 = "sora.Elementwise"(%78, %79) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %81:6 = "sora.Split"(%80) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %82 = "sora.Layernorm"(%76) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %83 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %84 = "sora.Elementwise"(%81#1, %83) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %85 = "sora.Elementwise"(%82, %84) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %86 = "sora.Elementwise"(%85, %81#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %87 = "sora.View"(%86) <{shape = [2, 28, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<2x28x405x1152xsi8>
    %88 = "sora.Transpose"(%87) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x28x405x1152xsi8>) -> tensor<2x405x28x1152xsi8>
    %89 = "sora.View"(%88) <{shape = [810, 28, 1152]}> : (tensor<2x405x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %90 = "sora.Weight"() : () -> tensor<1152xf16>
    %91 = "sora.LinearW8"(%89, %26, %90) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<362880x72xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %92 = "sora.LinearW8"(%89, %28, %29) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xsi8>, tensor<362880x405xf16>) -> tensor<22680x1152xsi8>
    %93 = "sora.LinearW8"(%89, %30, %31) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<362880x405xf16>, tensor<64512x405xf16>) -> tensor<22680x1152xsi8>
    %94 = "sora.View"(%91) <{shape = [810, 28, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x16x72xf16>
    %95 = "sora.View"(%92) <{shape = [810, 28, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x16x72xf16>
    %96 = "sora.View"(%93) <{shape = [810, 28, 16, 72]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x16x72xf16>
    %97 = "sora.Transpose"(%94) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<362880x72xf16>
    %98 = "sora.Transpose"(%95) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<362880x72xf16>
    %99 = "sora.Transpose"(%96) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %100 = "sora.Rmsnorm"(%97) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %101 = "sora.Rmsnorm"(%98) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %102 = "sora.Rope"(%100, %33) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<362880x72xf16>, tensor<56x16x405x72xf16>) -> tensor<810x16x28x72xf16>
    %103 = "sora.Rope"(%101, %33) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<362880x72xf16>, tensor<56x16x405x72xf16>) -> tensor<362880x72xf16>
    %104 = "sora.Convert"(%103) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %105 = "sora.Weight"() : () -> tensor<810x16x28x72xf16>
    %106 = "sora.Elementwise"(%102, %105) <{dynamic_scale = true, op_type = "div"}> : (tensor<810x16x28x72xf16>, tensor<810x16x28x72xf16>) -> tensor<810x16x28x72xsi8>
    %107 = "sora.MatmulW8"(%106, %104) : (tensor<810x16x28x72xsi8>, tensor<362880x72xf16>) -> tensor<362880x28xf16>
    %108 = "sora.Softmax"(%107) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x28xf16>) -> tensor<362880x28xf16>
    %109 = "sora.Transpose"(%99) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<933120x28xf16>
    %110 = "sora.Convert"(%109) <{dynamic_scale = true}> : (tensor<933120x28xf16>) -> tensor<933120x28xf16>
    %111 = "sora.MatmulW8"(%108, %110) : (tensor<362880x28xf16>, tensor<933120x28xf16>) -> tensor<810x16x28x72xf16>
    %112 = "sora.Transpose"(%111) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<810x28x16x72xf16>
    %113 = "sora.View"(%112) <{shape = [810, 28, 1152]}> : (tensor<810x28x16x72xf16>) -> tensor<22680x1152xf16>
    %114 = "sora.Convert"(%113) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %115 = "sora.LinearW8"(%114, %34, %36) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<56x405x16x72xf16>, tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %116 = "sora.View"(%115) <{shape = [2, 405, 28, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x405x28x1152xf16>
    %117 = "sora.Transpose"(%116) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x405x28x1152xf16>) -> tensor<2x28x405x1152xf16>
    %118 = "sora.View"(%117) <{shape = [2, 11340, 1152]}> : (tensor<2x28x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %119 = "sora.Elementwise"(%118, %81#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %120 = "sora.Elementwise"(%76, %119) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %121 = "sora.Convert"(%120) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %122 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %123 = "sora.LinearW8"(%121, %122, %38) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<1152x1152xsi8>, tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %124 = "sora.LinearW8"(%1, %39, %41) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<2x11340x1152xf16>, tensor<22680x1152xf16>) -> tensor<224x1152xf16>
    %125 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %126 = "sora.LinearW8"(%1, %125, %43) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<1152x1152xsi8>, tensor<224x1152xf16>) -> tensor<224x1152xf16>
    %127 = "sora.View"(%123) <{shape = [1, 22680, 16, 72]}> : (tensor<22680x1152xf16>) -> tensor<1x22680x16x72xf16>
    %128 = "sora.View"(%124) <{shape = [1, 224, 16, 72]}> : (tensor<224x1152xf16>) -> tensor<1x224x16x72xf16>
    %129 = "sora.View"(%126) <{shape = [1, 224, 16, 72]}> : (tensor<224x1152xf16>) -> tensor<1x224x16x72xf16>
    %130 = "sora.Transpose"(%127) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %131 = "sora.Transpose"(%128) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<3584x72xf16>
    %132 = "sora.Transpose"(%129) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %133 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %134 = "sora.Elementwise"(%130, %133) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %135 = "sora.Convert"(%131) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<3584x72xf16>
    %136 = "sora.MatmulW8"(%134, %135) : (tensor<1x16x22680x72xsi8>, tensor<3584x72xf16>) -> tensor<1x16x22680x224xf16>
    %137 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %138 = "sora.Elementwise"(%136, %137) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<362880x224xf16>
    %139 = "sora.Softmax"(%138) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<362880x224xf16>
    %140 = "sora.Transpose"(%132) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1152x224xf16>
    %141 = "sora.Convert"(%140) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1152x224xf16>
    %142 = "sora.MatmulW8"(%139, %141) : (tensor<362880x224xf16>, tensor<1152x224xf16>) -> tensor<1x16x22680x72xf16>
    %143 = "sora.Transpose"(%142) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %144 = "sora.View"(%143) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<22680x1152xf16>
    %145 = "sora.Convert"(%144) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %146 = "sora.LinearW8"(%145, %44, %46) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<224x1152xf16>, tensor<1x224x16x72xf16>) -> tensor<22680x1152xf16>
    %147 = "sora.Elementwise"(%120, %146) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %148 = "sora.Layernorm"(%147) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %149 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %150 = "sora.Elementwise"(%81#4, %149) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %151 = "sora.Elementwise"(%148, %150) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %152 = "sora.Elementwise"(%151, %81#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xsi8>
    %153 = "sora.LinearW8"(%152, %48, %50) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1x16x22680x72xf16>, tensor<1x16x224x72xf16>) -> tensor<22680x1152xsi8>
    %154 = "sora.Gelu"(%153) <{dynamic_scale = true}> : (tensor<22680x1152xsi8>) -> tensor<22680x1152xsi8>
    %155 = "sora.Weight"() : () -> tensor<1152x4608xsi8>
    %156 = "sora.Weight"() : () -> tensor<1152xf16>
    %157 = "sora.LinearW8"(%154, %155, %156) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x4608xsi8>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %158 = "sora.Elementwise"(%157, %81#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<22680x1152xsi8>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %159 = "sora.Elementwise"(%147, %158) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %160 = "sora.Elementwise"(%146, %158) <{dynamic_scale = false, op_type = "add"}> : (tensor<22680x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    return %160 : tensor<2x11340x1152xf16>
  }
}

