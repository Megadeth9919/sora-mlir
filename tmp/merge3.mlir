module @Sora {
  func.func @main(%arg0: tensor<2x11340x1152xf16>, %arg1: tensor<1x224x1152xf16>, %arg2: tensor<2x6912xf16>) -> tensor<2x11340x1152xf16> {
    %0 = "sora.None"() : () -> none
    %1 = "sora.View"(%arg1) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %2 = "sora.Convert"(%1) <{dynamic_scale = true}> : (tensor<224x1152xf16>) -> tensor<224x1152xf16>
    %3 = "sora.View"(%2) <{shape = [1, 224, 1152]}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xf16>
    %4 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %5 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %6 = "sora.Elementwise"(%4, %5) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %7:6 = "sora.Split"(%6) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %8 = "sora.View"(%arg0) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %9 = "sora.Layernorm"(%8) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %10 = "sora.View"(%9) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %11 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %12 = "sora.Elementwise"(%7#1, %11) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %13 = "sora.Elementwise"(%10, %12) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %14 = "sora.Elementwise"(%13, %7#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %15 = "sora.View"(%14) <{shape = [56, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<56x405x1152xsi8>
    %16 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %17 = "sora.Weight"() : () -> tensor<1152xf16>
    %18 = "sora.View"(%15) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %19 = "sora.LinearW8"(%18, %16, %17) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x1152xsi8>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %20 = "sora.View"(%19) <{shape = [56, 405, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x1152xsi8>
    %21 = "sora.Weight"() : () -> tensor<1152xf16>
    %22 = "sora.View"(%15) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %23 = "sora.LinearW8"(%22, %arg0, %21) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x11340x1152xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %24 = "sora.View"(%23) <{shape = [56, 405, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x1152xsi8>
    %25 = "sora.Weight"() : () -> tensor<1152xf16>
    %26 = "sora.View"(%15) <{shape = [22680, 1152]}> : (tensor<56x405x1152xsi8>) -> tensor<22680x1152xsi8>
    %27 = "sora.LinearW8"(%26, %arg2, %25) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<2x6912xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %28 = "sora.View"(%27) <{shape = [56, 405, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<56x405x1152xsi8>
    %29 = "sora.View"(%20) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xsi8>) -> tensor<56x405x16x72xf16>
    %30 = "sora.View"(%24) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xsi8>) -> tensor<56x405x16x72xf16>
    %31 = "sora.View"(%28) <{shape = [56, 405, 16, 72]}> : (tensor<56x405x1152xsi8>) -> tensor<56x405x16x72xf16>
    %32 = "sora.Transpose"(%29) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %33 = "sora.Transpose"(%30) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %34 = "sora.Transpose"(%31) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x405x16x72xf16>) -> tensor<56x16x405x72xf16>
    %35 = "sora.View"(%32) <{shape = [362880, 72]}> : (tensor<56x16x405x72xf16>) -> tensor<362880x72xf16>
    %36 = "sora.Rmsnorm"(%35) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %37 = "sora.View"(%36) <{shape = [56, 16, 405, 72]}> : (tensor<362880x72xf16>) -> tensor<56x16x405x72xf16>
    %38 = "sora.View"(%33) <{shape = [362880, 72]}> : (tensor<56x16x405x72xf16>) -> tensor<362880x72xf16>
    %39 = "sora.Rmsnorm"(%38) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %40 = "sora.View"(%39) <{shape = [56, 16, 405, 72]}> : (tensor<362880x72xf16>) -> tensor<56x16x405x72xf16>
    %41 = "sora.Weight"() : () -> tensor<56x16x405x72xf16>
    %42 = "sora.Elementwise"(%37, %41) <{dynamic_scale = true, op_type = "div"}> : (tensor<56x16x405x72xf16>, tensor<56x16x405x72xf16>) -> tensor<56x16x405x72xsi8>
    %43 = "sora.MatmulW8"(%42, %40) : (tensor<56x16x405x72xsi8>, tensor<56x16x405x72xf16>) -> tensor<56x16x405x405xf16>
    %44 = "sora.View"(%43) <{shape = [362880, 405]}> : (tensor<56x16x405x405xf16>) -> tensor<362880x405xf16>
    %45 = "sora.Softmax"(%44) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x405xf16>) -> tensor<362880x405xf16>
    %46 = "sora.View"(%45) <{shape = [56, 16, 405, 405]}> : (tensor<362880x405xf16>) -> tensor<56x16x405x405xf16>
    %47 = "sora.Transpose"(%34) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<56x16x72x405xf16>
    %48 = "sora.View"(%47) <{shape = [64512, 405]}> : (tensor<56x16x72x405xf16>) -> tensor<64512x405xf16>
    %49 = "sora.Convert"(%48) <{dynamic_scale = true}> : (tensor<64512x405xf16>) -> tensor<64512x405xf16>
    %50 = "sora.View"(%49) <{shape = [56, 16, 72, 405]}> : (tensor<64512x405xf16>) -> tensor<56x16x72x405xf16>
    %51 = "sora.MatmulW8"(%46, %50) : (tensor<56x16x405x405xf16>, tensor<56x16x72x405xf16>) -> tensor<56x16x405x72xf16>
    %52 = "sora.Transpose"(%51) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<56x16x405x72xf16>) -> tensor<56x405x16x72xf16>
    %53 = "sora.View"(%52) <{shape = [56, 405, 1152]}> : (tensor<56x405x16x72xf16>) -> tensor<56x405x1152xf16>
    %54 = "sora.View"(%53) <{shape = [22680, 1152]}> : (tensor<56x405x1152xf16>) -> tensor<22680x1152xf16>
    %55 = "sora.Convert"(%54) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %56 = "sora.View"(%55) <{shape = [56, 405, 1152]}> : (tensor<22680x1152xf16>) -> tensor<56x405x1152xf16>
    %57 = "sora.View"(%56) <{shape = [22680, 1152]}> : (tensor<56x405x1152xf16>) -> tensor<22680x1152xf16>
    %58 = "sora.LinearW8"(%57, %7#0, %7#2) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xf16>
    %59 = "sora.View"(%58) <{shape = [56, 405, 1152]}> : (tensor<22680x1152xf16>) -> tensor<56x405x1152xf16>
    %60 = "sora.View"(%59) <{shape = [2, 11340, 1152]}> : (tensor<56x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %61 = "sora.Elementwise"(%60, %7#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %62 = "sora.Elementwise"(%arg0, %61) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %63 = "sora.View"(%62) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %64 = "sora.Convert"(%63) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %65 = "sora.View"(%64) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %66 = "sora.View"(%65) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %67 = "sora.LinearW8"(%66, %7#3, %7#5) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<22680x1152xf16>
    %68 = "sora.View"(%67) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %69 = "sora.View"(%3) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %70 = "sora.LinearW8"(%69, %10, %13) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<224x1152xf16>
    %71 = "sora.View"(%70) <{shape = [1, 224, 1152]}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xf16>
    %72 = "sora.View"(%3) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %73 = "sora.LinearW8"(%72, %14, %15) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<2x11340x1152xsi8>, tensor<56x405x1152xsi8>) -> tensor<224x1152xf16>
    %74 = "sora.View"(%73) <{shape = [1, 224, 1152]}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xf16>
    %75 = "sora.View"(%68) <{shape = [1, 22680, 16, 72]}> : (tensor<2x11340x1152xf16>) -> tensor<1x22680x16x72xf16>
    %76 = "sora.View"(%71) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %77 = "sora.View"(%74) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %78 = "sora.Transpose"(%75) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %79 = "sora.Transpose"(%76) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %80 = "sora.Transpose"(%77) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %81 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %82 = "sora.Elementwise"(%78, %81) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %83 = "sora.View"(%79) <{shape = [3584, 72]}> : (tensor<1x16x224x72xf16>) -> tensor<3584x72xf16>
    %84 = "sora.Convert"(%83) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<3584x72xf16>
    %85 = "sora.View"(%84) <{shape = [1, 16, 224, 72]}> : (tensor<3584x72xf16>) -> tensor<1x16x224x72xf16>
    %86 = "sora.MatmulW8"(%82, %85) : (tensor<1x16x22680x72xsi8>, tensor<1x16x224x72xf16>) -> tensor<1x16x22680x224xf16>
    %87 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %88 = "sora.Elementwise"(%86, %87) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<1x16x22680x224xf16>
    %89 = "sora.View"(%88) <{shape = [362880, 224]}> : (tensor<1x16x22680x224xf16>) -> tensor<362880x224xf16>
    %90 = "sora.Softmax"(%89) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<362880x224xf16>
    %91 = "sora.View"(%90) <{shape = [1, 16, 22680, 224]}> : (tensor<362880x224xf16>) -> tensor<1x16x22680x224xf16>
    %92 = "sora.Transpose"(%80) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1x16x72x224xf16>
    %93 = "sora.View"(%92) <{shape = [1152, 224]}> : (tensor<1x16x72x224xf16>) -> tensor<1152x224xf16>
    %94 = "sora.Convert"(%93) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1152x224xf16>
    %95 = "sora.View"(%94) <{shape = [1, 16, 72, 224]}> : (tensor<1152x224xf16>) -> tensor<1x16x72x224xf16>
    %96 = "sora.MatmulW8"(%91, %95) : (tensor<1x16x22680x224xf16>, tensor<1x16x72x224xf16>) -> tensor<1x16x22680x72xf16>
    %97 = "sora.Transpose"(%96) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %98 = "sora.View"(%97) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<2x11340x1152xf16>
    %99 = "sora.View"(%98) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %100 = "sora.Convert"(%99) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %101 = "sora.View"(%100) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %102 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %103 = "sora.View"(%101) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %104 = "sora.LinearW8"(%103, %102, %24) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<1152x1152xsi8>, tensor<56x405x1152xsi8>) -> tensor<22680x1152xf16>
    %105 = "sora.View"(%104) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %106 = "sora.Elementwise"(%62, %105) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %107 = "sora.View"(%106) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %108 = "sora.Layernorm"(%107) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %109 = "sora.View"(%108) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %110 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %111 = "sora.Elementwise"(%7#4, %110) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %112 = "sora.Elementwise"(%109, %111) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %113 = "sora.Elementwise"(%112, %7#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %114 = "sora.View"(%113) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %115 = "sora.LinearW8"(%114, %29, %31) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x405x16x72xf16>, tensor<56x405x16x72xf16>) -> tensor<22680x1152xsi8>
    %116 = "sora.View"(%115) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %117 = "sora.View"(%116) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %118 = "sora.Gelu"(%117) <{dynamic_scale = true}> : (tensor<22680x1152xsi8>) -> tensor<22680x1152xsi8>
    %119 = "sora.View"(%118) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %120 = "sora.View"(%119) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %121 = "sora.LinearW8"(%120, %32, %34) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xf16>, tensor<56x16x405x72xf16>) -> tensor<22680x1152xsi8>
    %122 = "sora.View"(%121) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %123 = "sora.Elementwise"(%122, %7#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xsi8>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %124 = "sora.Elementwise"(%106, %123) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %125 = "sora.Elementwise"(%105, %123) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %126 = "sora.View"(%arg2) <{shape = [2, 6, 1152]}> : (tensor<2x6912xf16>) -> tensor<2x6x1152xf16>
    %127 = "sora.Weight"() : () -> tensor<1x6x1152xf16>
    %128 = "sora.Elementwise"(%126, %127) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x6x1152xf16>, tensor<1x6x1152xf16>) -> tensor<2x6x1152xf16>
    %129:6 = "sora.Split"(%128) <{dim = 1 : si32, split_size = 6 : si32}> : (tensor<2x6x1152xf16>) -> (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>, tensor<2x1x1152xf16>)
    %130 = "sora.View"(%124) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %131 = "sora.Layernorm"(%130) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %132 = "sora.View"(%131) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %133 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %134 = "sora.Elementwise"(%129#1, %133) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %135 = "sora.Elementwise"(%132, %134) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %136 = "sora.Elementwise"(%135, %129#0) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %137 = "sora.View"(%136) <{shape = [2, 28, 405, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<2x28x405x1152xsi8>
    %138 = "sora.Transpose"(%137) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x28x405x1152xsi8>) -> tensor<2x405x28x1152xsi8>
    %139 = "sora.View"(%138) <{shape = [810, 28, 1152]}> : (tensor<2x405x28x1152xsi8>) -> tensor<810x28x1152xsi8>
    %140 = "sora.Weight"() : () -> tensor<1152xf16>
    %141 = "sora.View"(%139) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %142 = "sora.LinearW8"(%141, %40, %140) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xf16>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %143 = "sora.View"(%142) <{shape = [810, 28, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x1152xsi8>
    %144 = "sora.View"(%139) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %145 = "sora.LinearW8"(%144, %42, %43) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x72xsi8>, tensor<56x16x405x405xf16>) -> tensor<22680x1152xsi8>
    %146 = "sora.View"(%145) <{shape = [810, 28, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x1152xsi8>
    %147 = "sora.View"(%139) <{shape = [22680, 1152]}> : (tensor<810x28x1152xsi8>) -> tensor<22680x1152xsi8>
    %148 = "sora.LinearW8"(%147, %46, %47) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<56x16x405x405xf16>, tensor<56x16x72x405xf16>) -> tensor<22680x1152xsi8>
    %149 = "sora.View"(%148) <{shape = [810, 28, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<810x28x1152xsi8>
    %150 = "sora.View"(%143) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xsi8>) -> tensor<810x28x16x72xf16>
    %151 = "sora.View"(%146) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xsi8>) -> tensor<810x28x16x72xf16>
    %152 = "sora.View"(%149) <{shape = [810, 28, 16, 72]}> : (tensor<810x28x1152xsi8>) -> tensor<810x28x16x72xf16>
    %153 = "sora.Transpose"(%150) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %154 = "sora.Transpose"(%151) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %155 = "sora.Transpose"(%152) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x28x16x72xf16>) -> tensor<810x16x28x72xf16>
    %156 = "sora.View"(%153) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %157 = "sora.Rmsnorm"(%156) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %158 = "sora.View"(%157) <{shape = [810, 16, 28, 72]}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xf16>
    %159 = "sora.View"(%154) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %160 = "sora.Rmsnorm"(%159) <{dynamic_scale = false}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %161 = "sora.View"(%160) <{shape = [810, 16, 28, 72]}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xf16>
    %162 = "sora.Rope"(%158, %51) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<810x16x28x72xf16>, tensor<56x16x405x72xf16>) -> tensor<810x16x28x72xf16>
    %163 = "sora.Rope"(%161, %51) <{dim = 1152 : si32, dynamic_scale = false}> : (tensor<810x16x28x72xf16>, tensor<56x16x405x72xf16>) -> tensor<810x16x28x72xf16>
    %164 = "sora.View"(%163) <{shape = [362880, 72]}> : (tensor<810x16x28x72xf16>) -> tensor<362880x72xf16>
    %165 = "sora.Convert"(%164) <{dynamic_scale = true}> : (tensor<362880x72xf16>) -> tensor<362880x72xf16>
    %166 = "sora.View"(%165) <{shape = [810, 16, 28, 72]}> : (tensor<362880x72xf16>) -> tensor<810x16x28x72xf16>
    %167 = "sora.Weight"() : () -> tensor<810x16x28x72xf16>
    %168 = "sora.Elementwise"(%162, %167) <{dynamic_scale = true, op_type = "div"}> : (tensor<810x16x28x72xf16>, tensor<810x16x28x72xf16>) -> tensor<810x16x28x72xsi8>
    %169 = "sora.MatmulW8"(%168, %166) : (tensor<810x16x28x72xsi8>, tensor<810x16x28x72xf16>) -> tensor<810x16x28x28xf16>
    %170 = "sora.View"(%169) <{shape = [362880, 28]}> : (tensor<810x16x28x28xf16>) -> tensor<362880x28xf16>
    %171 = "sora.Softmax"(%170) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x28xf16>) -> tensor<362880x28xf16>
    %172 = "sora.View"(%171) <{shape = [810, 16, 28, 28]}> : (tensor<362880x28xf16>) -> tensor<810x16x28x28xf16>
    %173 = "sora.Transpose"(%155) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<810x16x72x28xf16>
    %174 = "sora.View"(%173) <{shape = [933120, 28]}> : (tensor<810x16x72x28xf16>) -> tensor<933120x28xf16>
    %175 = "sora.Convert"(%174) <{dynamic_scale = true}> : (tensor<933120x28xf16>) -> tensor<933120x28xf16>
    %176 = "sora.View"(%175) <{shape = [810, 16, 72, 28]}> : (tensor<933120x28xf16>) -> tensor<810x16x72x28xf16>
    %177 = "sora.MatmulW8"(%172, %176) : (tensor<810x16x28x28xf16>, tensor<810x16x72x28xf16>) -> tensor<810x16x28x72xf16>
    %178 = "sora.Transpose"(%177) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<810x16x28x72xf16>) -> tensor<810x28x16x72xf16>
    %179 = "sora.View"(%178) <{shape = [810, 28, 1152]}> : (tensor<810x28x16x72xf16>) -> tensor<810x28x1152xf16>
    %180 = "sora.View"(%179) <{shape = [22680, 1152]}> : (tensor<810x28x1152xf16>) -> tensor<22680x1152xf16>
    %181 = "sora.Convert"(%180) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %182 = "sora.View"(%181) <{shape = [810, 28, 1152]}> : (tensor<22680x1152xf16>) -> tensor<810x28x1152xf16>
    %183 = "sora.View"(%182) <{shape = [22680, 1152]}> : (tensor<810x28x1152xf16>) -> tensor<22680x1152xf16>
    %184 = "sora.LinearW8"(%183, %52, %56) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<56x405x16x72xf16>, tensor<56x405x1152xf16>) -> tensor<22680x1152xf16>
    %185 = "sora.View"(%184) <{shape = [810, 28, 1152]}> : (tensor<22680x1152xf16>) -> tensor<810x28x1152xf16>
    %186 = "sora.View"(%185) <{shape = [2, 405, 28, 1152]}> : (tensor<810x28x1152xf16>) -> tensor<2x405x28x1152xf16>
    %187 = "sora.Transpose"(%186) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<2x405x28x1152xf16>) -> tensor<2x28x405x1152xf16>
    %188 = "sora.View"(%187) <{shape = [2, 11340, 1152]}> : (tensor<2x28x405x1152xf16>) -> tensor<2x11340x1152xf16>
    %189 = "sora.Elementwise"(%188, %129#2) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %190 = "sora.Elementwise"(%124, %189) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %191 = "sora.View"(%190) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %192 = "sora.Convert"(%191) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %193 = "sora.View"(%192) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %194 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %195 = "sora.View"(%193) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %196 = "sora.LinearW8"(%195, %194, %60) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<1152x1152xsi8>, tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %197 = "sora.View"(%196) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %198 = "sora.View"(%3) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %199 = "sora.LinearW8"(%198, %61, %65) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<224x1152xf16>
    %200 = "sora.View"(%199) <{shape = [1, 224, 1152]}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xf16>
    %201 = "sora.Weight"() : () -> tensor<1152x1152xsi8>
    %202 = "sora.View"(%3) <{shape = [224, 1152]}> : (tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %203 = "sora.LinearW8"(%202, %201, %71) <{do_bias = true}> : (tensor<224x1152xf16>, tensor<1152x1152xsi8>, tensor<1x224x1152xf16>) -> tensor<224x1152xf16>
    %204 = "sora.View"(%203) <{shape = [1, 224, 1152]}> : (tensor<224x1152xf16>) -> tensor<1x224x1152xf16>
    %205 = "sora.View"(%197) <{shape = [1, 22680, 16, 72]}> : (tensor<2x11340x1152xf16>) -> tensor<1x22680x16x72xf16>
    %206 = "sora.View"(%200) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %207 = "sora.View"(%204) <{shape = [1, 224, 16, 72]}> : (tensor<1x224x1152xf16>) -> tensor<1x224x16x72xf16>
    %208 = "sora.Transpose"(%205) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x22680x16x72xf16>) -> tensor<1x16x22680x72xf16>
    %209 = "sora.Transpose"(%206) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %210 = "sora.Transpose"(%207) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x224x16x72xf16>) -> tensor<1x16x224x72xf16>
    %211 = "sora.Weight"() : () -> tensor<1x16x22680x72xf16>
    %212 = "sora.Elementwise"(%208, %211) <{dynamic_scale = true, op_type = "div"}> : (tensor<1x16x22680x72xf16>, tensor<1x16x22680x72xf16>) -> tensor<1x16x22680x72xsi8>
    %213 = "sora.View"(%209) <{shape = [3584, 72]}> : (tensor<1x16x224x72xf16>) -> tensor<3584x72xf16>
    %214 = "sora.Convert"(%213) <{dynamic_scale = true}> : (tensor<3584x72xf16>) -> tensor<3584x72xf16>
    %215 = "sora.View"(%214) <{shape = [1, 16, 224, 72]}> : (tensor<3584x72xf16>) -> tensor<1x16x224x72xf16>
    %216 = "sora.MatmulW8"(%212, %215) : (tensor<1x16x22680x72xsi8>, tensor<1x16x224x72xf16>) -> tensor<1x16x22680x224xf16>
    %217 = "sora.Weight"() : () -> tensor<1x1x22680x224xf16>
    %218 = "sora.Elementwise"(%216, %217) <{dynamic_scale = false, op_type = "add"}> : (tensor<1x16x22680x224xf16>, tensor<1x1x22680x224xf16>) -> tensor<1x16x22680x224xf16>
    %219 = "sora.View"(%218) <{shape = [362880, 224]}> : (tensor<1x16x22680x224xf16>) -> tensor<362880x224xf16>
    %220 = "sora.Softmax"(%219) <{dim = -1 : si32, dynamic_scale = true}> : (tensor<362880x224xf16>) -> tensor<362880x224xf16>
    %221 = "sora.View"(%220) <{shape = [1, 16, 22680, 224]}> : (tensor<362880x224xf16>) -> tensor<1x16x22680x224xf16>
    %222 = "sora.Transpose"(%210) <{dim_a = 3 : si32, dim_b = 2 : si32}> : (tensor<1x16x224x72xf16>) -> tensor<1x16x72x224xf16>
    %223 = "sora.View"(%222) <{shape = [1152, 224]}> : (tensor<1x16x72x224xf16>) -> tensor<1152x224xf16>
    %224 = "sora.Convert"(%223) <{dynamic_scale = true}> : (tensor<1152x224xf16>) -> tensor<1152x224xf16>
    %225 = "sora.View"(%224) <{shape = [1, 16, 72, 224]}> : (tensor<1152x224xf16>) -> tensor<1x16x72x224xf16>
    %226 = "sora.MatmulW8"(%221, %225) : (tensor<1x16x22680x224xf16>, tensor<1x16x72x224xf16>) -> tensor<1x16x22680x72xf16>
    %227 = "sora.Transpose"(%226) <{dim_a = 1 : si32, dim_b = 2 : si32}> : (tensor<1x16x22680x72xf16>) -> tensor<1x22680x16x72xf16>
    %228 = "sora.View"(%227) <{shape = [2, 11340, 1152]}> : (tensor<1x22680x16x72xf16>) -> tensor<2x11340x1152xf16>
    %229 = "sora.View"(%228) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %230 = "sora.Convert"(%229) <{dynamic_scale = true}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %231 = "sora.View"(%230) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %232 = "sora.View"(%231) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %233 = "sora.LinearW8"(%232, %74, %76) <{do_bias = true}> : (tensor<22680x1152xf16>, tensor<1x224x1152xf16>, tensor<1x224x16x72xf16>) -> tensor<22680x1152xf16>
    %234 = "sora.View"(%233) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %235 = "sora.Elementwise"(%190, %234) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %236 = "sora.View"(%235) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xf16>) -> tensor<22680x1152xf16>
    %237 = "sora.Layernorm"(%236) <{dynamic_scale = false}> : (tensor<22680x1152xf16>) -> tensor<22680x1152xf16>
    %238 = "sora.View"(%237) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xf16>) -> tensor<2x11340x1152xf16>
    %239 = "sora.Weight"() : () -> tensor<2x1x1152xf16>
    %240 = "sora.Elementwise"(%129#4, %239) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x1x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x1x1152xf16>
    %241 = "sora.Elementwise"(%238, %240) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %242 = "sora.Elementwise"(%241, %129#3) <{dynamic_scale = true, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xsi8>
    %243 = "sora.View"(%242) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %244 = "sora.LinearW8"(%243, %78, %80) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1x16x22680x72xf16>, tensor<1x16x224x72xf16>) -> tensor<22680x1152xsi8>
    %245 = "sora.View"(%244) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %246 = "sora.View"(%245) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %247 = "sora.Gelu"(%246) <{dynamic_scale = true}> : (tensor<22680x1152xsi8>) -> tensor<22680x1152xsi8>
    %248 = "sora.View"(%247) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %249 = "sora.Weight"() : () -> tensor<1152x4608xsi8>
    %250 = "sora.Weight"() : () -> tensor<1152xf16>
    %251 = "sora.View"(%248) <{shape = [22680, 1152]}> : (tensor<2x11340x1152xsi8>) -> tensor<22680x1152xsi8>
    %252 = "sora.LinearW8"(%251, %249, %250) <{do_bias = true}> : (tensor<22680x1152xsi8>, tensor<1152x4608xsi8>, tensor<1152xf16>) -> tensor<22680x1152xsi8>
    %253 = "sora.View"(%252) <{shape = [2, 11340, 1152]}> : (tensor<22680x1152xsi8>) -> tensor<2x11340x1152xsi8>
    %254 = "sora.Elementwise"(%253, %129#5) <{dynamic_scale = false, op_type = "mul"}> : (tensor<2x11340x1152xsi8>, tensor<2x1x1152xf16>) -> tensor<2x11340x1152xf16>
    %255 = "sora.Elementwise"(%235, %254) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    %256 = "sora.Elementwise"(%234, %254) <{dynamic_scale = false, op_type = "add"}> : (tensor<2x11340x1152xf16>, tensor<2x11340x1152xf16>) -> tensor<2x11340x1152xf16>
    return %256 : tensor<2x11340x1152xf16>
  }
}

