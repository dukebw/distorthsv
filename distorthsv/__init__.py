# Copyright 2018 Brendan Duke.
#
# This file is part of distorthsv.
#
# distorthsv is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# distorthsv is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# distorthsv. If not, see <http://www.gnu.org/licenses/>.

"""Wrapper for the distorthsv C extension APIs."""
import _distorthsv


distorthsv = _distorthsv.distorthsv
distort_contrast = _distorthsv.distort_contrast
fliplr = _distorthsv.fliplr
