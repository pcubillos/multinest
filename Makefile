# Makefile - prepared for C multinest package
#
# `make` - Build and compile the pyratbay package.
# `make clean` - Remove all compiled (non-source) files that are created.
#
# If you are interested in the commands being run by this makefile, you may add
# "VERBOSE=1" to the end of any `make` command, i.e.:
#
# 		make VERBOSE=1
#
# This will display the exact commands being used for building, etc.

# Set verbosity
#
Q = @
O = > /dev/null
ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
		O =
	endif
else
	MAKEFLAGS += --no-print-directory
endif

LIBDIR = lib/

# Get the location of this Makefile.
mkfile_dir := $(dir $(lastword $(MAKEFILE_LIST)))

# `make [clean]` should run `make [clean]` on all of the modules.
all: make_pb
clean: clean_pb


make_pb:
	@echo "Building multinest nested-sampling package."
	$(Q) python setup.py build $(O)
	@mv -f build/lib.*/lib/*.so $(LIBDIR)
	@rm -rf build/
	@echo "Successful compilation."
	@echo ""

clean_pb:
	@rm -rf $(LIBDIR)*.so
	@echo "Cleaned multinest package."

