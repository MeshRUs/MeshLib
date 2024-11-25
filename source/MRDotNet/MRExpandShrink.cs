using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Runtime.InteropServices;

namespace MR.DotNet
{    
    public class ExpandShrink
    {        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrExpandFaceRegion( IntPtr top, IntPtr region, int hops );
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrExpandFaceRegionFromFace( IntPtr top, FaceId face, int hops );
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrExpandVertRegion( IntPtr top, IntPtr region, int hops );
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrExpandVertRegionFromVert( IntPtr top, VertId vert, int hops );
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrShrinkFaceRegion( IntPtr top, IntPtr region, int hops );
       
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrShrinkVertRegion( IntPtr top, IntPtr region, int hops );
        
        /// adds to the region all faces within given number of hops (stars) from the initial region boundary
        public static void Expand( Mesh mesh, FaceBitSet region, int hops )
        {
            mrExpandFaceRegion( mesh.meshTopology_, region.bs_, hops );
        }
        /// returns the region of all faces within given number of hops (stars) from the initial face
        public static FaceBitSet Expand( Mesh mesh, FaceId face, int hops )
        {
            IntPtr res = mrExpandFaceRegionFromFace( mesh.meshTopology_, face, hops );
            return new FaceBitSet( res );
        }
        // adds to the region all vertices within given number of hops (stars) from the initial region boundary
        public static void Expand( Mesh mesh, VertBitSet region, int hops )
        {
            mrExpandVertRegion( mesh.meshTopology_, region.bs_, hops );
        }
        /// returns the region of all vertices within given number of hops (stars) from the initial vertex
        public static VertBitSet Expand( Mesh mesh, VertId vert, int hops )
        {
            IntPtr res = mrExpandVertRegionFromVert( mesh.meshTopology_, vert, hops );
            return new VertBitSet( res );
        }
        /// removes from the region all faces within given number of hops (stars) from the initial region boundary
        public static void Shrink( Mesh mesh, FaceBitSet region, int hops )
        {
            mrShrinkFaceRegion( mesh.meshTopology_, region.bs_, hops );
        }
        /// removes from the region all vertices within given number of hops (stars) from the initial region boundary
        public static void Shrink( Mesh mesh, VertBitSet region, int hops )
        {
            mrShrinkVertRegion( mesh.meshTopology_, region.bs_, hops );
        }
    }
}
