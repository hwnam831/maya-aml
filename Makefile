BUILDDIR=Build
DISTDIR=Dist
BALLOONDIR=Balloon

# Environment
MKDIR=mkdir
CP=cp
CC=gcc
CCC=g++
CXX=g++

# Project Name
PROJECTNAME=Maya

# Active Configuration
DEFAULTCONF=Release


# All Configurations
ALLCONFS=Debug Release 
CONF=${DEFAULTCONF}

# Libtorch related
TORCHDIR=${HOME}/libtorch
MYDEPS=-Wl,-rpath,${TORCHDIR}/lib ${TORCHDIR}/lib/libtorch.so ${TORCHDIR}/lib/libc10.so ${TORCHDIR}/lib/libkineto.a -Wl,--no-as-needed,"${TORCHDIR}/lib/libtorch_cpu.so" -Wl,--as-needed ${TORCHDIR}/lib/libc10.so -lpthread -Wl,--no-as-needed,"${TORCHDIR}/lib/libtorch.so" -Wl,--as-needed
CXX_DEFINES = -DUSE_C10D_GLOO -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE
CXX_INCLUDES = -isystem ${TORCHDIR}/include -isystem ${TORCHDIR}/include/torch/csrc/api/include
CXX_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=1 -std=gnu++14

# build
build: .validate-impl .depcheck-impl
	@#echo "=> Running $@... Configuration=$(CONF)"
	"${MAKE}" -f Makefile-${CONF}.mk QMAKE=${QMAKE} .build-conf

# clean
clean: .validate-impl .depcheck-impl
	@#echo "=> Running $@... Configuration=$(CONF)"
	"${MAKE}" -f Makefile-${CONF}.mk QMAKE=${QMAKE}  .clean-conf


# clobber
clobber: .depcheck-impl
	@#echo "=> Running $@..."
	for CONF in ${ALLCONFS}; \
	do \
	    "${MAKE}" -f Makefile-$${CONF}.mk QMAKE=${QMAKE} .clean-conf; \
	done

# all
all: .depcheck-impl
	@#echo "=> Running $@..."
	for CONF in ${ALLCONFS}; \
	do \
	    "${MAKE}" -f Makefile-$${CONF}.mk QMAKE=${QMAKE} .build-conf; \
	done


# help
help: 
	@echo "This makefile supports the following configurations:"
	@echo "    ${ALLCONFS}"
	@echo ""
	@echo "and the following targets:"
	@echo "    build  (default target)"
	@echo "    clean"
	@echo "    clobber"
	@echo "    all"
	@echo "    help"
	@echo ""
	@echo "Makefile Usage:"
	@echo "    make [CONF=<CONFIGURATION>] build"
	@echo "    make [CONF=<CONFIGURATION>] clean"
	@echo "    make clobber"
	@echo "    make all"
	@echo "    make help"
	@echo ""
	@echo "Target 'build' will build a specific configuration."
	@echo "Target 'clean' will clean a specific configuration."
	@echo "Target 'clobber' will remove all built files from all configurations."
	@echo "Target 'all' will will build all configurations."
	@echo "Target 'help' prints this message."
	@echo ""

# dependency checking support
.depcheck-impl:
	@echo "# This code depends on make tool being used" >.dep.inc
	@if [ -n "${MAKE_VERSION}" ]; then \
	    echo "DEPFILES=\$$(wildcard \$$(addsuffix .d, \$${OBJECTFILES} \$${BALLOONOBJ} \$${TESTOBJECTFILES}))" >>.dep.inc; \
	    echo "ifneq (\$${DEPFILES},)" >>.dep.inc; \
	    echo "include \$${DEPFILES}" >>.dep.inc; \
	    echo "endif" >>.dep.inc; \
	else \
	    echo ".KEEP_STATE:" >>.dep.inc; \
	    echo ".KEEP_STATE_FILE:.make.state.\$${CONF}" >>.dep.inc; \
	fi

# configuration validation
.validate-impl:
	@if [ ! -f Makefile-${CONF}.mk ]; \
	then \
	    echo ""; \
	    echo "Error: can not find the makefile for configuration '${CONF}' in project ${PROJECTNAME}"; \
	    echo "See 'make help' for details."; \
	    echo "Current directory: " `pwd`; \
	    echo ""; \
	fi
	@if [ ! -f Makefile-${CONF}.mk ]; \
	then \
	    exit 1; \
	fi
