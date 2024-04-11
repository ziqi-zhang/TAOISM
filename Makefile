######## SGX SDK Settings ########

SGX_SDK ?= /home/zhangziqi/sgx-2.15/sgxsdk

SGX_MODE ?= HW
SGX_ARCH ?= x64
SGX_DEBUG ?= 0
SGX_PRERELEASE ?= 1

# SGX_MODE ?= SIM
# SGX_ARCH ?= x64
# SGX_DEBUG ?= 1
# SGX_PRERELEASE ?= 0

ifeq ($(shell getconf LONG_BIT), 32)
	SGX_ARCH := x86
else ifeq ($(findstring -m32, $(CXXFLAGS)), -m32)
	SGX_ARCH := x86
endif

ifeq ($(SGX_ARCH), x86)
	SGX_COMMON_CFLAGS := -m32 -w 
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x86/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x86/sgx_edger8r
else
	SGX_COMMON_CFLAGS := -m64 -w
	SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
	SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
	SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r
endif

ifeq ($(SGX_DEBUG), 1)
ifeq ($(SGX_PRERELEASE), 1)
$(error Cannot set SGX_DEBUG and SGX_PRERELEASE at the same time!!)
endif
endif

ifeq ($(SGX_DEBUG), 1)
        SGX_COMMON_CFLAGS += -O3 -g
else
        SGX_COMMON_CFLAGS += -O3
endif

######## App Settings ########

ifneq ($(SGX_MODE), HW)
	Urts_Library_Name := sgx_urts_sim
else
	Urts_Library_Name := sgx_urts
endif

App_CC_Files :=
App_Cpp_Files := App/enclave_bridge.cpp
App_Include_Paths := -IApp -I$(SGX_SDK)/include -I$(TF_INC) -I$(TF_INC)/external/nsync/public -I./Include -I./Include/eigen # -I./SGXDNN 

App_C_Flags := $(SGX_COMMON_CFLAGS) -fPIC -Wno-attributes $(App_Include_Paths) -fopenmp -march=native -maes -ffast-math -ftree-vectorize

# Three configuration modes - Debug, prerelease, release
#   Debug - Macro DEBUG enabled.
#   Prerelease - Macro NDEBUG and EDEBUG enabled.
#   Release - Macro NDEBUG enabled.
ifeq ($(SGX_DEBUG), 1)
        App_C_Flags += -DDEBUG -UNDEBUG -UEDEBUG
else ifeq ($(SGX_PRERELEASE), 1)
        App_C_Flags += -DNDEBUG -DEDEBUG -UDEBUG
else
        App_C_Flags += -DNDEBUG -UEDEBUG -UDEBUG
endif

App_Cpp_Flags_WithoutSGX := $(App_C_Flags) -std=c++11 -shared 
App_Cpp_Flags := $(App_Cpp_Flags_WithoutSGX) -DUSE_SGX
App_Link_Flags := $(SGX_COMMON_CFLAGS) -L$(SGX_LIBRARY_PATH) -l$(Urts_Library_Name) -L$(TF_LIB) -pthread -fopenmp

ifneq ($(SGX_MODE), HW)
	App_Link_Flags += -lsgx_uae_service_sim -pthread
else
	App_Link_Flags += -lsgx_uae_service
endif

App_CC_Objects := $(App_CC_Files:.cc=.so)
App_Cpp_Objects := $(App_Cpp_Files:.cpp=.so)
App_Name := taoism_app

######## Enclave Settings ########

ifneq ($(SGX_MODE), HW)
	Trts_Library_Name := sgx_trts_sim
	Service_Library_Name := sgx_tservice_sim
else
	Trts_Library_Name := sgx_trts
	Service_Library_Name := sgx_tservice
endif
Crypto_Library_Name := sgx_tcrypto

Enclave_Cpp_Files := Enclave/Enclave.cpp Enclave/sgxdnn.cpp 
SGXDNN_Cpp_Files := sgxdnn_main.cpp chunk_manager.cpp secret_tensor.cpp xoshiro.cpp stochastic.cpp Crypto.cpp sgxaes.cpp aes-stream.cpp utils.cpp 
SGXDNN_Layer_Cpp_Files := layers/batchnorm.cpp layers/linear.cpp layers/conv.cpp layers/maxpool.cpp
#SGXDNN_Cpp_Files += aesni_ghash.cpp aesni_key.cpp  aesni-wrap.cpp
Enclave_Include_Paths := -IEnclave -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx
Enclave_Include_Paths += -IInclude -ISGXDNN -IInclude/eigen3_sgx -I/usr/lib/gcc/x86_64-linux-gnu/7.5.0/include

CC_BELOW_4_9 := $(shell expr "`$(CC) -dumpversion`" \< "4.9")
ifeq ($(CC_BELOW_4_9), 1)
	Enclave_C_Flags := $(SGX_COMMON_CFLAGS) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -fstack-protector
else
	Enclave_C_Flags := $(SGX_COMMON_CFLAGS) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -fstack-protector-strong
endif

Enclave_C_Flags += $(Enclave_Include_Paths)
Enclave_Cpp_Flags := $(Enclave_C_Flags) -std=c++11 -nostdinc++ -DUSE_SGX -DEIGEN_NO_CPUID
Enclave_Cpp_Flags += -march=native -maes -ffast-math

# To generate a proper enclave, it is recommended to follow below guideline to link the trusted libraries:
#    1. Link sgx_trts with the `--whole-archive' and `--no-whole-archive' options,
#       so that the whole content of trts is included in the enclave.
#    2. For other libraries, you just need to pull the required symbols.
#       Use `--start-group' and `--end-group' to link these libraries.
# Do NOT move the libraries linked with `--start-group' and `--end-group' within `--whole-archive' and `--no-whole-archive' options.
# Otherwise, you may get some undesirable errors.
Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
	-Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -lsgx_omp -lsgx_pthread -l$(Crypto_Library_Name) -l$(Service_Library_Name) -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
	-Wl,--defsym,__ImageBase=0 -Wl,--gc-sections   \
	-Wl,--version-script=Enclave/Enclave.lds

Enclave_Cpp_Objects := $(Enclave_Cpp_Files:Enclave/%.cpp=Enclave/bin/%.o)
SGXDNN_Cpp_Objects := $(SGXDNN_Cpp_Files:%.cpp=SGXDNN/bin_sgx/%.o)
SGXDNN_Layer_Cpp_Objects := $(SGXDNN_Layer_Cpp_Files:%.cpp=SGXDNN/bin_sgx/%.o)

Enclave_Name := enclave.so
Signed_Enclave_Name := enclave.signed.so
Enclave_Config_File := Enclave/Enclave.config.xml

ifeq ($(SGX_MODE), HW)
ifeq ($(SGX_DEBUG), 1)
	Build_Mode = HW_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	Build_Mode = HW_PRERELEASE
else
	Build_Mode = HW_RELEASE
endif
else
ifeq ($(SGX_DEBUG), 1)
	Build_Mode = SIM_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	Build_Mode = SIM_PRERELEASE
else
	Build_Mode = SIM_RELEASE
endif
endif


.PHONY: all run

ifeq ($(Build_Mode), HW_RELEASE)
all: .config_$(Build_Mode)_$(SGX_ARCH) $(App_Name) $(Enclave_Name)
	@echo "The project has been built in release hardware mode."
	@echo "Please sign the $(Enclave_Name) first with your signing key before you run the $(App_Name) to launch and access the enclave."
	@echo "To sign the enclave use the command:"
	@echo "   $(SGX_ENCLAVE_SIGNER) sign -key <your key> -enclave $(Enclave_Name) -out <$(Signed_Enclave_Name)> -config $(Enclave_Config_File)"
	@echo "You can also sign the enclave using an external signing tool."
	@echo "To build the project in simulation mode set SGX_MODE=SIM. To build the project in prerelease mode set SGX_PRERELEASE=1 and SGX_MODE=HW."
else
all: .config_$(Build_Mode)_$(SGX_ARCH) $(App_Name) $(Signed_Enclave_Name)
ifeq ($(Build_Mode), HW_DEBUG)
	@echo "The project has been built in debug hardware mode."
else ifeq ($(Build_Mode), SIM_DEBUG)
	@echo "The project has been built in debug simulation mode."
else ifeq ($(Build_Mode), HW_PRERELEASE)
	@echo "The project has been built in pre-release hardware mode."
else ifeq ($(Build_Mode), SIM_PRERELEASE)
	@echo "The project has been built in pre-release simulation mode."
else
	@echo "The project has been built in release simulation mode."
endif
endif

run: all
#ifneq ($(Build_Mode), HW_RELEASE)
#	@$(CURDIR)/$(App_Name)
#	@echo "RUN  =>  $(App_Name) [$(SGX_MODE)|$(SGX_ARCH), OK]"
#endif

######## App Objects ########

App/Enclave_u.c: $(SGX_EDGER8R) Enclave/Enclave.edl
	cd App && mkdir -p bin/ && $(SGX_EDGER8R) --untrusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include 
	@echo "GEN  =>  $@  current---------------dir $(pwd)"

App/bin/Enclave_u.o: App/Enclave_u.c
	$(CC) $(App_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

App/bin/aes_stream_common.o: App/aes_stream_common.cpp
	$(CXX) $(App_Cpp_Flags_WithoutSGX) App/aes_stream_common.cpp -o $@ $(App_Link_Flags)
	@echo "CXX  <=  $<"

App/bin/sgxaes_common.o: App/sgxaes_common.cpp SGXDNN/bin_sgx/sgxaes_asm.o 
	$(CXX) $(App_Cpp_Flags_WithoutSGX) App/sgxaes_common.cpp -o $@ SGXDNN/bin_sgx/sgxaes_asm.o $(App_Link_Flags)
	@echo "CXX  <=  $<"

App/bin/randpool_common.o: App/randpool_common.cpp App/bin/aes_stream_common.o 
	$(CXX) $(App_Cpp_Flags_WithoutSGX) App/randpool_common.cpp -o $@ App/bin/aes_stream_common.o $(App_Link_Flags)
	@echo "CXX  <=  $<"

App/bin/common_utils.o: App/common_utils.cpp
	$(CXX) $(App_Cpp_Flags_WithoutSGX) App/randpool_common.cpp -o $@ $(App_Link_Flags)
	@echo "CXX  <=  $<"


App/bin/crypto_common.o: App/crypto_common.cpp App/bin/sgxaes_common.o
	$(CXX) $(App_Cpp_Flags_WithoutSGX) App/crypto_common.cpp -o $@ App/bin/sgxaes_common.o $(App_Link_Flags)
	@echo "CXX  <=  $<"

App/bin/enclave_bridge.so: App/enclave_bridge.cpp App/bin/Enclave_u.o App/bin/aes_stream_common.o App/bin/crypto_common.o App/bin/randpool_common.o App/bin/common_utils.o
	$(CXX) $(App_Cpp_Flags) App/enclave_bridge.cpp -o $@ \
		App/bin/Enclave_u.o App/bin/aes_stream_common.o App/bin/sgxaes_common.o App/bin/crypto_common.o App/bin/randpool_common.o App/bin/common_utils.o \
		$(App_Link_Flags)
	@echo "CXX  <=  $<"

$(App_Name): App/bin/Enclave_u.o App/bin/enclave_bridge.so #$(App_Cpp_Objects) $(App_CC_Objects)
	$(CXX) $^ -o $@ $(App_Link_Flags)
	@echo "LINK =>  $@"

.config_$(Build_Mode)_$(SGX_ARCH):
	#rm -f .config_* $(App_Cpp_Objects) $(App_CC_Objects)
	rm -f .config_* App/enclave_bridge.so 
	@touch .config_$(Build_Mode)_$(SGX_ARCH)

######## Enclave Objects ########

Enclave/Enclave_t.c: $(SGX_EDGER8R) Enclave/Enclave.edl
	cd Enclave && mkdir -p bin/ && $(SGX_EDGER8R) --trusted ../Enclave/Enclave.edl --search-path ../Enclave --search-path $(SGX_SDK)/include
	@echo "GEN  =>  $@"

Enclave/bin/Enclave_t.o: Enclave/Enclave_t.c
	$(CC) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC   <=  $<"

SGXDNN/bin_sgx/%.o: SGXDNN/%.S
	mkdir -p SGXDNN/bin_sgx
	$(CC) -c $< -o $@
	@echo "CC   <=  $<"

Enclave/bin/%.o: Enclave/%.cpp
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

# SGXDNN/aes_stream_common.o: App/aes_stream_common.cpp
# 	$(CXX) $(Enclave_Cpp_Flags) SGXDNN/aes_stream_common.cpp -o $@
# 	@echo "CXX  <=  $<"

SGXDNN/bin_sgx/%.o: SGXDNN/%.cpp
	mkdir -p SGXDNN/bin_sgx
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"


SGXDNN/bin_sgx/layers/%.o: SGXDNN/layers/%.cpp
	mkdir -p SGXDNN/bin_sgx/layers
	$(CXX) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX  <=  $<"

$(Enclave_Name): Enclave/bin/Enclave_t.o $(Enclave_Cpp_Objects) $(SGXDNN_Cpp_Objects) $(SGXDNN_Layer_Cpp_Objects) SGXDNN/bin_sgx/sgxaes_asm.o 
	@echo "BEFORE LINK =--------------------------------->  $@"
	$(CXX) $^ -o $@ $(Enclave_Link_Flags)
	@echo "LINK =>  $@"

$(Signed_Enclave_Name): $(Enclave_Name)
	$(SGX_ENCLAVE_SIGNER) sign -key Enclave/Enclave_private.pem -enclave $(Enclave_Name) -out $@ -config $(Enclave_Config_File)
	@echo "SIGN =>  $@"

.PHONY: clean

clean:
	rm -f .config_* $(Enclave_Name) $(Signed_Enclave_Name) $(App_Cpp_Objects) $(App_CC_Objects) App/Enclave_u.* 
	rm -f App/aes_stream_common.o App/crypto_common.o App/sgxaes_common.o
	rm -f $(Enclave_Cpp_Objects) $(SGXDNN_Cpp_Objects) $(SGXDNN_Cpp_Objects) Enclave/Enclave_t.* SGXDNN/bin_sgx/*.o

	rm -rf App/bin App/Enclave_u.*
	rm -rf Enclave/bin Enclave/Enclave_t.*
