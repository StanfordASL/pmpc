# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Mosek_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Mosek")
JLLWrappers.@generate_main_file("Mosek", UUID("87c66717-d3e3-50e4-a91b-58f286718dc5"))
end  # module Mosek_jll
