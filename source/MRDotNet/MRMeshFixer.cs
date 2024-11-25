using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    [StructLayout(LayoutKind.Sequential)]
    public struct MultipleEdge
    {
        VertId v0;
        VertId v1;
    }

    public class MeshFixer
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindHoleComplicatingFaces(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindDegenerateFaces( ref MRMeshPart mp, float criticalAspectRatio, IntPtr cb, ref IntPtr errorStr);
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindShortEdges(ref MRMeshPart mp, float criticalLength, IntPtr cb, ref IntPtr errorStr);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void fixMultipleEdges(IntPtr mesh, MultipleEdge* multipleEdges, ulong multipleEdgesNum );
        
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void findAndFixMultipleEdges(IntPtr mesh);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrStringData(IntPtr str);

        /// returns all faces that complicate one of mesh holes;
        /// hole is complicated if it passes via one vertex more than once;
        /// deleting such faces simplifies the holes and makes them easier to fill
        public static BitSet FindHoleComplicatingFaces(Mesh mesh)
        {
            return new BitSet(mrFindHoleComplicatingFaces(mesh.mesh_));
        }
        /// finds faces having aspect ratio >= criticalAspectRatio
        public static FaceBitSet FindDegenerateFaces(MeshPart meshPart, float criticalAspectRatio)
        {
            IntPtr errorString = new IntPtr();
            IntPtr res = mrFindDegenerateFaces(ref meshPart.mrMeshPart, criticalAspectRatio, IntPtr.Zero, ref errorString);
            if (errorString != IntPtr.Zero)
            {
                var errData = mrStringData(errorString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new Exception(errorMessage);
            }
            return new FaceBitSet(res);
        }
        /// finds edges having length <= criticalLength
        public static UndirectedEdgeBitSet FindShortEdges(MeshPart meshPart, float criticalLength)
        {
            IntPtr errorString = new IntPtr();
            IntPtr res = mrFindShortEdges(ref meshPart.mrMeshPart, criticalLength, IntPtr.Zero, ref errorString);
            if (errorString != IntPtr.Zero)
            {
                var errData = mrStringData(errorString);
                string errorMessage = Marshal.PtrToStringAnsi(errData);
                throw new Exception(errorMessage);
            }
            return new UndirectedEdgeBitSet(res);
        }
        /// resolves given multiple edges, but splitting all but one edge in each group
        unsafe public static void FixMultipleEdges( ref Mesh mesh, List<MultipleEdge> multipleEdges)
        {
            MultipleEdge* multipleEdgesPtr = stackalloc MultipleEdge[multipleEdges.Count];
            for (int i = 0; i < multipleEdges.Count; i++)
                multipleEdgesPtr[i] = multipleEdges[i];

            fixMultipleEdges(mesh.mesh_, multipleEdgesPtr, (ulong)multipleEdges.Count);
        }
        /// finds and resolves multiple edges
        public static void FixMultipleEdges( ref Mesh mesh )
        {
            findAndFixMultipleEdges(mesh.mesh_);
        }
    }
}
