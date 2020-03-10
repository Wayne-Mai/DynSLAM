/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#pragma once

#include "SegmentationResult.h"
#include "../FrameData.h"

class GPUTexture;

class SegmentationPerformer {
 public:

    virtual SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models,
                                              FrameDataPointer frame,
                                              unsigned char nextModelID,
                                              bool allowNew) = 0;
    virtual ~SegmentationPerformer() = default;

    virtual std::vector<std::pair<std::string, std::shared_ptr<GPUTexture>>> getDrawableTextures() { return {}; }

    inline void setNewModelMinRelativeSize(float v) { minRelSizeNew = v; }
    inline void setNewModelMaxRelativeSize(float v) { maxRelSizeNew = v; }

 protected:
    // post-processing
    float maxRelSizeNew = 0.4;
    float minRelSizeNew = 0.07;
};
