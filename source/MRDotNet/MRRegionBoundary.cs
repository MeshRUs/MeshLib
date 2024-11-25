using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace MR.DotNet
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct MREdgeLoop
    {
        public IntPtr data = IntPtr.Zero;
        public ulong size = 0;
        public IntPtr reserved = IntPtr.Zero;

        public MREdgeLoop() { }
    }

    public class EdgeLoop : List<EdgeId>, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void mrEdgePathFree(MREdgeLoop* loop);

        unsafe internal EdgeLoop(MREdgeLoop* mrLoop)
        : base()
        {
            mrLoop_ = mrLoop;
            for ( int i = 0; i < (int)mrLoop->size; i++ )
            {
                Add(new EdgeId(Marshal.ReadInt32(IntPtr.Add(mrLoop->data, i * sizeof(int)))));
            }
        }
        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        unsafe protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrLoop_ != null)
                {
                    mrEdgePathFree(mrLoop_);
                }

                disposed = true;
            }
        }

        ~EdgeLoop()
        {
            Dispose(false);
        }

        unsafe internal MREdgeLoop* mrLoop_;
    }

    public class EdgeLoops : List<List<EdgeId>>, IDisposable
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern MREdgeLoop mrEdgeLoopsGet(IntPtr loops, ulong index);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern ulong mrEdgeLoopsSize(IntPtr loops);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern void mrEdgeLoopsFree(IntPtr loops);

        public EdgeLoops( IntPtr mrLoops )
        : base()
        {
            mrLoops_ = mrLoops;
            int numLoops = (int)mrEdgeLoopsSize(mrLoops);
            for ( int i = 0; i < numLoops; i++ )
            {
                Add(new List<EdgeId>());
                var mrLoop = mrEdgeLoopsGet(mrLoops, (ulong)i);
                for ( int j = 0; j < (int)mrLoop.size; j++ )
                {
                    this[i].Add(new EdgeId(Marshal.ReadInt32(IntPtr.Add(mrLoop.data, j * sizeof(int)))));
                }
            }
        }

        private bool disposed = false;
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        unsafe protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (mrLoops_ != null)
                {
                    mrEdgeLoopsFree(mrLoops_);
                }

                disposed = true;
            }
        }

        ~EdgeLoops()
        {
            Dispose(false);
        }

        internal IntPtr mrLoops_;
    }

    public class RegionBoundary
    {
        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrFindRightBoundary( IntPtr topology, IntPtr region );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern MREdgeLoop* mrTrackRightBoundaryLoop(IntPtr topology, EdgeId e0, IntPtr region);

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentFacesFromVerts( IntPtr topology, IntPtr region );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentFacesFromEdges( IntPtr topology, IntPtr region );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentVertsFromFaces( IntPtr topology, IntPtr faces );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentVertsFromEdges( IntPtr topology, IntPtr edges );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetInnerVertsFromFaces( IntPtr topology, IntPtr region );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetInnerVertsFromEdges( IntPtr topology, IntPtr edges );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetInnerFacesFromVerts( IntPtr topology, IntPtr verts );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentEdgesFromFaces( IntPtr topology, IntPtr faces );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetIncidentEdgesFromEdges( IntPtr topology, IntPtr edges );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetInnerEdgesFromVerts( IntPtr topology, IntPtr verts );

        [DllImport("MRMeshC.dll", CharSet = CharSet.Ansi)]
        private static extern IntPtr mrGetInnerEdgesFromFaces( IntPtr topology, IntPtr region );

        /// returns closed loop of region boundary starting from given region boundary edge (region faces on the right, and not-region faces or holes on the left);
        /// if more than two boundary edges connect in one vertex, then the function makes the most abrupt turn to left
        unsafe public static EdgeLoop TrackRightBoundaryLoop( Mesh mesh, EdgeId e0, FaceBitSet? region = null )
        {
            var mrLoop = mrTrackRightBoundaryLoop(mesh.meshTopology_, e0, region is null ? (IntPtr)null : region.bs_);
            return new EdgeLoop(mrLoop);
        }
        /// returns all region boundary loops;
        /// every loop has region faces on the right, and not-region faces or holes on the left
        public static EdgeLoops FindRightBoundary( Mesh mesh, FaceBitSet? region = null )
        {
            var mrLoops = mrFindRightBoundary(mesh.meshTopology_, region is null ? (IntPtr)null : region.bs_);
            return new EdgeLoops(mrLoops);
        }
        /// composes the set of all faces incident to given vertices
        public static FaceBitSet GetIncidentFaces(Mesh mesh, VertBitSet region)
        {
            return new FaceBitSet(mrGetIncidentFacesFromVerts(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all faces incident to given edges
        public static FaceBitSet GetIncidentFaces(Mesh mesh, UndirectedEdgeBitSet region)
        {
            return new FaceBitSet(mrGetIncidentFacesFromEdges(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all vertices incident to given faces
        public static VertBitSet GetIncidentVerts(Mesh mesh, FaceBitSet region)
        {
            return new VertBitSet(mrGetIncidentVertsFromFaces(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all vertices incident to given edges
        public static VertBitSet GetIncidentVerts(Mesh mesh, UndirectedEdgeBitSet region)
        {
            return new VertBitSet(mrGetIncidentVertsFromEdges(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all vertices not on the boundary of a hole and with all their adjacent faces in given set
        public static VertBitSet GetInnerVerts(Mesh mesh, FaceBitSet region)
        {
            return new VertBitSet(mrGetInnerVertsFromFaces(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all vertices with all their edges in given set
        public static VertBitSet GetInnerVerts(Mesh mesh, UndirectedEdgeBitSet region)
        {
            return new VertBitSet(mrGetInnerVertsFromEdges(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all faces with all their vertices in given set
        public static FaceBitSet GetInnerFaces(Mesh mesh, VertBitSet region)
        {
            return new FaceBitSet(mrGetInnerFacesFromVerts(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all edges with all their vertices in given set
        public static UndirectedEdgeBitSet GetInnerEdges(Mesh mesh, VertBitSet region)
        {
            return new UndirectedEdgeBitSet(mrGetInnerEdgesFromVerts(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all edges having both left and right in given region
        public static UndirectedEdgeBitSet GetInnerEdges(Mesh mesh, FaceBitSet region)
        {
            return new UndirectedEdgeBitSet(mrGetInnerEdgesFromFaces(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all undirected edges, having a face from given set from one of two sides
        public static UndirectedEdgeBitSet GetIncidentEdges(Mesh mesh, FaceBitSet region)
        {
            return new UndirectedEdgeBitSet(mrGetIncidentEdgesFromFaces(mesh.meshTopology_, region.bs_));
        }
        /// composes the set of all undirected edges, having at least one common vertex with an edge from given set
        public static UndirectedEdgeBitSet GetIncidentEdges(Mesh mesh, UndirectedEdgeBitSet region)
        {
            return new UndirectedEdgeBitSet(mrGetIncidentEdgesFromEdges(mesh.meshTopology_, region.bs_));
        }
    }
}
