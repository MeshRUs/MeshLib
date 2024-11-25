using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;
using System.Text;
using static MR.DotNet.MeshComponents;

namespace MR.DotNet
{
    public struct MeshRegions
    {
        public FaceBitSet faces;
        public int numRegions = 0;

        public MeshRegions(FaceBitSet faces, int numRegions)
        {
            this.faces = faces;
            this.numRegions = numRegions;
        }
    };

    public enum FaceIncidence
    {
        PerEdge, /// face can have neighbor only via edge
        PerVertex /// face can have neighbor via vertex
    };

    public class MeshComponentsMap : List<RegionId>, IDisposable
    {
        internal MRMeshComponentsMap mrMap_;
        private bool disposed_ = false;

        public int NumComponents = 0;

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern void mrMeshComponentsAllComponentsMapFree(MRMeshComponentsMap* map);

        unsafe internal MeshComponentsMap(MRMeshComponentsMap mrMap)
            : base()
        {
            mrMap_ = mrMap;
            NumComponents = mrMap.numComponents;

            for (int i = 0; i < (int)mrMap.faceMap->size; i++)
            {
                Add(new RegionId(Marshal.ReadInt32(IntPtr.Add(mrMap.faceMap->data, i * sizeof(int)))));
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        unsafe protected virtual void Dispose(bool disposing)
        {
            if (!disposed_)
            {
                if (mrMap_.faceMap->data != IntPtr.Zero)
                    fixed ( MRMeshComponentsMap* p = &mrMap_ ) mrMeshComponentsAllComponentsMapFree(p);

                disposed_ = true;
            }
        }

        ~MeshComponentsMap()
        {
            Dispose(false);
        }
    }

    public class MeshComponents
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct MRFace2RegionMap
        {
            public IntPtr data = IntPtr.Zero;
            public ulong size = 0;
            public IntPtr reserved = IntPtr.Zero;
            public MRFace2RegionMap() { }
        }

        [StructLayout(LayoutKind.Sequential)]
        unsafe internal struct MRMeshComponentsMap
        {
            public MRFace2RegionMap* faceMap = null;
            public int numComponents = 0;
            public MRMeshComponentsMap() {}
        };

        [StructLayout(LayoutKind.Sequential)]
        internal struct MRMeshRegions
        {
            public IntPtr faces = IntPtr.Zero;
            public int numRegions = 0;
            public MRMeshRegions() { }
        };

        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshComponentsGetComponent(ref MRMeshPart mp, FaceId id, FaceIncidence incidence, IntPtr cb);

        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern IntPtr mrMeshComponentsGetLargestComponent(ref MRMeshPart mp, FaceIncidence incidence, IntPtr cb, float minArea, int* numSmallerComponents);
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern IntPtr mrMeshComponentsGetLargeByAreaComponents(ref MRMeshPart mp, float minArea, IntPtr cb);

        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        private static extern MRMeshComponentsMap mrMeshComponentsGetAllComponentsMap(ref MRMeshPart mp, FaceIncidence incidence);

        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Auto)]
        unsafe private static extern MRMeshRegions mrMeshComponentsGetLargeByAreaRegions( ref MRMeshPart mp, MRFace2RegionMap* face2RegionMap, int numRegions, float minArea );

        /// gets all connected components of mesh part as
        /// 1. the mapping: FaceId -> Component ID in [0, 1, 2, ...)
        /// 2. the total number of components
        unsafe static public MeshComponentsMap GetAllComponentsMap(MeshPart mp, FaceIncidence incidence)
        {
            var mrMap = mrMeshComponentsGetAllComponentsMap(ref mp.mrMeshPart, incidence);
            var res = new MeshComponentsMap(mrMap);
            return res;
        }
        /// returns
        /// 1. the union of all regions with area >= minArea
        /// 2. the number of such regions
        unsafe static public MeshRegions GetLargeByAreaRegions(MeshPart mp, MeshComponentsMap map, int numRegions, float minArea)
        {
            var mrRegions = mrMeshComponentsGetLargeByAreaRegions(ref mp.mrMeshPart, map.mrMap_.faceMap, numRegions, minArea);
            return new MeshRegions
            {
                faces = new FaceBitSet(mrRegions.faces),
                numRegions = mrRegions.numRegions
            };
        }
        /// returns the union of connected components, each having at least given area
        static public FaceBitSet GetLargeByAreaComponents( MeshPart mp, float minArea )
        {
            var components = mrMeshComponentsGetLargeByAreaComponents(ref mp.mrMeshPart, minArea, IntPtr.Zero);
            return new FaceBitSet(components);
        }
        /// returns the largest by surface area component or empty set if its area is smaller than \param minArea        
        unsafe static public FaceBitSet GetLargestComponent( MeshPart mp, FaceIncidence incidence, float minArea, out int numSmallerComponents )
        {
            fixed (int* p = &numSmallerComponents) 
                return new FaceBitSet( mrMeshComponentsGetLargestComponent(ref mp.mrMeshPart, incidence, IntPtr.Zero, minArea, p) );
        }
        /// not effective to call more than once, if several components are needed use GetAllComponentsMap
        static public FaceBitSet GetComponent( MeshPart mp, FaceId id, FaceIncidence incidence )
        {
            return new FaceBitSet( mrMeshComponentsGetComponent(ref mp.mrMeshPart, id, incidence, IntPtr.Zero) );
        }
    }
}
